import gym
import time
from contextlib import contextmanager
import random
import stl
from stl import mesh
import tempfile
import os
import numpy as np

from shutil import copyfile, copy2

# find the max dimensions, so we can know the bounding box, getting the height,
# width, length (because these are the step size)...
def find_mins_maxs(obj):
    minx = maxx = miny = maxy = minz = maxz = None
    for p in obj.points:
        # p contains (x, y, z)
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    return minx, maxx, miny, maxy, minz, maxz

def default_model(name, regen_fn=None):
    """
    Get a model with basic settings such as gravity and RK4 integration enabled
    """
    model = MJCModelRegen(name, regen_fn)
    root = model.root

    # Setup
    root.compiler(angle="radian", inertiafromgeom="true")
    default = root.default()
    default.joint(armature=1, damping=1, limited="true")
    default.geom(contype=0, friction='1 0.1 0.1', rgba='0.7 0.7 0 1')
    root.option(gravity="0 0 -9.81", integrator="RK4", timestep=0.01)
    return model

def pointmass_model(name):
    """
    Get a model with basic settings such as gravity and Euler integration enabled
    """
    model = MJCModel(name)
    root = model.root

    # Setup
    root.compiler(angle="radian", inertiafromgeom="true", coordinate="local")
    default = root.default()
    default.joint(limited="false", damping=1)
    default.geom(contype=2, conaffinity="1", condim="1", friction=".5 .1 .1", density="1000", margin="0.002")
    root.option(timestep=0.01, gravity="0 0 0", iterations="20", integrator="Euler")
    return model


class MJCModel(object):
    def __init__(self, name):
        self.name = name
        self.root = MJCTreeNode("mujoco").add_attr('model', name)

    @contextmanager
    def asfile(self):
        """
        Usage:

        model = MJCModel('reacher')
        with model.asfile() as f:
            print f.read()  # prints a dump of the model

        """
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.xml', delete=True) as f:
            self.root.write(f)
            f.seek(0)
            yield f

    def open(self):
        self.file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.xml', delete=True)
        self.root.write(self.file)
        self.file.seek(0)
        return self.file

    def save(self, path):
        with open(path, 'w') as f:
            self.root.write(f)

    def close(self):
        self.file.close()


class MJCModelRegen(MJCModel):
    def __init__(self, name, regen_fn):
        super(MJCModelRegen, self).__init__(name)
        self.regen_fn = regen_fn

    def regenerate(self):
        self.root = self.regen_fn().root



class MJCTreeNode(object):
    def __init__(self, name):
        self.name = name
        self.attrs = {}
        self.children = []

    def add_attr(self, key, value):
        if isinstance(value, str):  # should be basestring in python2
            pass
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            value = ' '.join([str(val) for val in value])

        self.attrs[key] = value
        return self

    def __getattr__(self, name):
        def wrapper(**kwargs):
            newnode =  MJCTreeNode(name)
            for (k, v) in kwargs.items(): # iteritems in python2
                newnode.add_attr(k, v)
            self.children.append(newnode)
            return newnode
        return wrapper

    def dfs(self):
        yield self
        if self.children:
            for child in self.children:
                for node in child.dfs():
                    yield node

    def write(self, ostream, tabs=0):
        contents = ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        if self.children:

            ostream.write('\t'*tabs)
            ostream.write('<%s %s>\n' % (self.name, contents))
            for child in self.children:
                child.write(ostream, tabs=tabs+1)
            ostream.write('\t'*tabs)
            ostream.write('</%s>\n' % self.name)
        else:
            ostream.write('\t'*tabs)
            ostream.write('<%s %s/>\n' % (self.name, contents))

    def __str__(self):
        s = "<"+self.name
        s += ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        return s+">"

def pusher(obj_scale=None,obj_mass=None,obj_damping=None,object_pos=(0.45, -0.05, -0.275),distr_scale=None,actual_distr_scale=None, distr_mass=None, distr_damping=None, goal_pos=(0.45, -0.05, -0.3230), distractor_pos=(0.45,-0.05,-0.275), N_objects=1, mesh_file=None,mesh_file_path=None, distractor_mesh_file=None, distractor_mesh_file_path=None, friction=(.8, .1, .1), table_texture=None, distractor_texture=None, obj_texture=None):
    object_pos, goal_pos, distractor_pos, friction = list(object_pos), list(goal_pos), list(distractor_pos), list(friction)
    # For now, only supports one distractor

    if distractor_mesh_file_path is None:
        distractor_mesh_file_path = distractor_mesh_file

    if obj_scale is None:
        obj_scale = random.uniform(0.5, 1.0)  # currently trying range of 0.5-1.0
    if obj_mass is None:
        obj_mass = random.uniform(0.1, 2.0)  # largest is 2.0, lowest is 0.1 I think
    if obj_damping is None:
        obj_damping = random.uniform(0.2, 5.0) # This is friction. ranges between 0.2 and 5.0
    obj_damping = str(obj_damping)

    if distractor_mesh_file:
        if distr_scale is None:
            # not used if actual_distr_scale is set.
            distr_scale = random.uniform(0.5, 1.0)  # currently trying range of 0.5-1.0
        if distr_mass is None:
            distr_mass = random.uniform(0.1, 2.0)  # largest is 2.0, lowest is 0.1 I think
        if distr_damping is None:
            distr_damping = random.uniform(0.2, 5.0) # This is friction. ranges between 0.2 and 5.0
        distr_damping = str(distr_damping)


    mjcmodel = MJCModel('arm3d')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01",gravity="0 0 0",iterations="20",integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(armature='0.04', damping=1, limited='true')
    default.geom(friction=friction,density="300",margin="0.002",condim="1",contype="0",conaffinity="0")

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(diffuse=".5 .5 .5", pos="0 0 3", dir="0 0 -1")
    if table_texture:
        worldbody.geom(name="table", material='table', type="plane", pos="0 0.5 -0.325", size="1 1 0.1", contype="1", conaffinity="1")
    else:
        worldbody.geom(name="table", type="plane", pos="0 0.5 -0.325", size="1 1 0.1", contype="1", conaffinity="1")
    r_shoulder_pan_link = worldbody.body(name="r_shoulder_pan_link", pos="0 -0.6 0")
    r_shoulder_pan_link.geom(name="e1", type="sphere", rgba="0.6 0.6 0.6 1", pos="-0.06 0.05 0.2", size="0.05")
    r_shoulder_pan_link.geom(name="e2", type="sphere", rgba="0.6 0.6 0.6 1", pos=" 0.06 0.05 0.2", size="0.05")
    r_shoulder_pan_link.geom(name="e1p", type="sphere", rgba="0.1 0.1 0.1 1", pos="-0.06 0.09 0.2", size="0.03")
    r_shoulder_pan_link.geom(name="e2p", type="sphere", rgba="0.1 0.1 0.1 1", pos=" 0.06 0.09 0.2", size="0.03")
    r_shoulder_pan_link.geom(name="sp", type="capsule", fromto="0 0 -0.4 0 0 0.2", size="0.1")
    r_shoulder_pan_link.joint(name="r_shoulder_pan_joint", type="hinge", pos="0 0 0", axis="0 0 1", range="-2.2854 1.714602", damping="1.0")
    r_shoulder_lift_link = r_shoulder_pan_link.body(name='r_shoulder_lift_link', pos="0.1 0 0")
    r_shoulder_lift_link.geom(name="sl", type="capsule", fromto="0 -0.1 0 0 0.1 0", size="0.1")
    r_shoulder_lift_link.joint(name="r_shoulder_lift_joint", type="hinge", pos="0 0 0", axis="0 1 0", range="-0.5236 1.3963", damping="1.0")
    r_upper_arm_roll_link = r_shoulder_lift_link.body(name="r_upper_arm_roll_link", pos="0 0 0")
    r_upper_arm_roll_link.geom(name="uar", type="capsule", fromto="-0.1 0 0 0.1 0 0", size="0.02")
    r_upper_arm_roll_link.joint(name="r_upper_arm_roll_joint", type="hinge", pos="0 0 0", axis="1 0 0", range="-1.5 1.7", damping="0.1")
    r_upper_arm_link = r_upper_arm_roll_link.body(name="r_upper_arm_link", pos="0 0 0")
    r_upper_arm_link.geom(name="ua", type="capsule", fromto="0 0 0 0.4 0 0", size="0.06")
    r_elbow_flex_link = r_upper_arm_link.body(name="r_elbow_flex_link", pos="0.4 0 0")
    r_elbow_flex_link.geom(name="ef", type="capsule", fromto="0 -0.02 0 0.0 0.02 0", size="0.06")
    r_elbow_flex_link.joint(name="r_elbow_flex_joint", type="hinge", pos="0 0 0", axis="0 1 0", range="-2.3213 0", damping="0.1")
    r_forearm_roll_link = r_elbow_flex_link.body(name="r_forearm_roll_link", pos="0 0 0")
    r_forearm_roll_link.geom(name="fr", type="capsule", fromto="-0.1 0 0 0.1 0 0", size="0.02")
    r_forearm_roll_link.joint(name="r_forearm_roll_joint", type="hinge", limited="true", pos="0 0 0", axis="1 0 0", damping=".1", range="-1.5 1.5")
    r_forearm_link = r_forearm_roll_link.body(name="r_forearm_link", pos="0 0 0")
    r_forearm_link.geom(name="fa", type="capsule", fromto="0 0 0 0.291 0 0", size="0.05")
    r_wrist_flex_link = r_forearm_link.body(name="r_wrist_flex_link", pos="0.321 0 0")
    r_wrist_flex_link.geom(name="wf", type="capsule", fromto="0 -0.02 0 0 0.02 0", size="0.01")
    r_wrist_flex_link.joint(name="r_wrist_flex_joint", type="hinge", pos="0 0 0", axis="0 1 0", range="-1.094 0", damping=".1")
    r_wrist_roll_link = r_wrist_flex_link.body(name="r_wrist_roll_link", pos="0 0 0")
    r_wrist_roll_link.joint(name="r_wrist_roll_joint", type="hinge", pos="0 0 0", limited="true", axis="1 0 0", damping="0.1", range="-1.5 1.5")
    tips_arm = r_wrist_roll_link.body(name="tips_arm", pos="0 0 0")
    tips_arm.geom(name="tip_arml", type="sphere", pos="0.1 -0.1 0.", size="0.01")
    tips_arm.geom(name="tip_armr", type="sphere", pos="0.1 0.1 0.", size="0.01")
    r_wrist_roll_link.geom(type="capsule", fromto="0 -0.1 0. 0.0 +0.1 0", size="0.02", contype="1", conaffinity="1")
    r_wrist_roll_link.geom(type="capsule", fromto="0 -0.1 0. 0.1 -0.1 0", size="0.02", contype="1", conaffinity="1")
    r_wrist_roll_link.geom(type="capsule", fromto="0 +0.1 0. 0.1 +0.1 0", size="0.02", contype="1", conaffinity="1")

    if mesh_file is not None:
        mesh_object = mesh.Mesh.from_file(mesh_file)
        vol, cog, inertia = mesh_object.get_mass_properties()
        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(mesh_object)
        max_length = max((maxx-minx),max((maxy-miny),(maxz-minz)))
        scale = obj_scale*0.0012 * (200.0 / max_length)
        object_density = obj_mass / (vol*scale*scale*scale)
        object_pos[0] -= scale*(minx+maxx)/2.0
        object_pos[1] -= scale*(miny+maxy)/2.0
        object_pos[2] = -0.324 - scale*minz
        object_scale = scale
    if distractor_mesh_file is not None:
        distr_mesh_object = mesh.Mesh.from_file(distractor_mesh_file)
        vol, cog, inertia = distr_mesh_object.get_mass_properties()
        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(distr_mesh_object)
        max_length = max((maxx-minx),max((maxy-miny),(maxz-minz)))
        if actual_distr_scale == None:
            actual_distr_scale = distr_scale*0.0012 * (200.0 / max_length)
        distr_density = distr_mass / (vol*actual_distr_scale*actual_distr_scale*actual_distr_scale)
        distractor_pos[0] -= actual_distr_scale*(minx+maxx)/2.0
        distractor_pos[1] -= actual_distr_scale*(miny+maxy)/2.0
        distractor_pos[2] = -0.324 - actual_distr_scale*minz

    ## MAKE DISTRACTOR
    if distractor_mesh_file:
        distractor = worldbody.body(name="distractor", pos=distractor_pos)#"0.45 -0.05 -0.275")
        #distractor.geom(rgba="1 1 1 0", type="sphere", size="0.05 0.05 0.05", density="0.00001", conaffinity="0")
        if distractor_mesh_file is None:
            distractor.geom(rgba="1 1 1 1", type="cylinder", size="0.05 0.05 0.05", density="0.00001", contype="1", conaffinity="0")
        else:
            # mesh = distractor.body(axisangle="1 0 0 1.57", pos="0 0 0") # axis angle might also need to be adjusted
            # TODO: do we need material here?
            if distractor_texture:
                distractor.geom(material='distractor', conaffinity="0", contype="1", density=str(distr_density), mesh="distractor_mesh" , rgba="1 1 1 1", type="mesh")
            else:
                distractor.geom(conaffinity="0", contype="1", density=str(distr_density), mesh="distractor_mesh" , rgba="1 1 1 1", type="mesh")
            # distal = mesh.body(name="distal_10_%d" % i, pos="0 0 0")
            # distal.site(name="distractor_pos_%d" % i, pos="0 0 0", size="0.01")
        distractor.joint(name="distractor_slidey", type="slide", pos="0 0 0", axis="0 1 0", range="-10.3213 10.3", damping=distr_damping)
        distractor.joint(name="distractor_slidex", type="slide", pos="0 0 0", axis="1 0 0", range="-10.3213 10.3", damping=distr_damping)

    # MAKE TARGET OBJECT
    object = worldbody.body(name="object", pos=object_pos)#"0.45 -0.05 -0.275")
    # object.geom(rgba="1 1 1 0", type="sphere", size="0.05 0.05 0.05", density="0.00001", conaffinity="0")
    if mesh_file is None:
        object.geom(rgba="1 1 1 1", type="cylinder", size="0.05 0.05 0.05", density="0.00001", contype="1", conaffinity="0")
    else:
        # mesh = object.body(axisangle="1 0 0 1.57", pos="0 0 0") # axis angle might also need to be adjusted
        # TODO: do we need material here?
        if obj_texture:
            object.geom(material='object', conaffinity="0", contype="1", density=str(object_density), mesh="object_mesh", rgba="1 1 1 1", type="mesh")
        else:
            object.geom(conaffinity="0", contype="1", density=str(object_density), mesh="object_mesh", rgba="1 1 1 1", type="mesh")
        # distal = mesh.body(name="distal_10", pos="0 0 0")
        # distal.site(name="obj_pos", pos="0 0 0", size="0.01")
    object.joint(name="obj_slidey", type="slide", pos="0 0 0", axis="0 1 0", range="-10.3213 10.3", damping=obj_damping)
    object.joint(name="obj_slidex", type="slide", pos="0 0 0", axis="1 0 0", range="-10.3213 10.3", damping=obj_damping)

    goal = worldbody.body(name="goal", pos=goal_pos)#"0.45 -0.05 -0.3230")
    goal.geom(rgba="1 0 0 1", type="cylinder", size="0.08 0.001 0.1", density='0.00001', contype="0", conaffinity="0")
    goal.joint(name="goal_slidey", type="slide", pos="0 0 0", axis="0 1 0", range="-10.3213 10.3", damping="0.5")
    goal.joint(name="goal_slidex", type="slide", pos="0 0 0", axis="1 0 0", range="-10.3213 10.3", damping="0.5")

    asset = mjcmodel.root.asset()
    if table_texture:
        asset.texture(name='table', file=table_texture, type='2d')
        asset.material(shininess='0.3', specular='1', name='table', rgba='0.9 0.9 0.9 1', texture='table')
    asset.mesh(file=mesh_file_path, name="object_mesh", scale=[object_scale]*3) # figure out the proper scale
    if distractor_mesh_file:
        asset.mesh(file=distractor_mesh_file_path, name="distractor_mesh", scale=[actual_distr_scale]*3)
        if distractor_texture:
            asset.texture(name='distractor', file=distractor_texture)
            asset.material(shininess='0.3', specular='1', name='distractor', rgba='0.9 0.9 0.9 1', texture='distractor')
    if obj_texture:
        asset.texture(name='object', file=obj_texture)
        asset.material(shininess='0.3', specular='1', name='object', rgba='0.9 0.9 0.9 1', texture='object')

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="r_shoulder_pan_joint", ctrlrange="-2.0 2.0", ctrllimited="true")
    actuator.motor(joint="r_shoulder_lift_joint", ctrlrange="-2.0 2.0", ctrllimited="true")
    actuator.motor(joint="r_upper_arm_roll_joint", ctrlrange="-2.0 2.0", ctrllimited="true")
    actuator.motor(joint="r_elbow_flex_joint", ctrlrange="-2.0 2.0", ctrllimited="true")
    actuator.motor(joint="r_forearm_roll_joint", ctrlrange="-2.0 2.0", ctrllimited="true")
    actuator.motor(joint="r_wrist_flex_joint", ctrlrange="-2.0 2.0", ctrllimited="true")
    actuator.motor(joint="r_wrist_roll_joint", ctrlrange="-2.0 2.0", ctrllimited="true")

    return mjcmodel

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Could edit this to be the path to the object file instead
    parser.add_argument('--xml_filepath', type=str, default='None')
    parser.add_argument('--obj_filepath', type=str, default='None')
    parser.add_argument('--debug_log', type=str, default='None')
    args = parser.parse_args()

    if args.debug_log != 'None':
        with open(args.debug_log, 'r') as f:
            i = 0
            mass = None
            scale = None
            obj = None
            damp = None
            xml_file = None
            for line in f:
                if 'scale:' in line:
                    # scale
                    string = line[line.index('scale:'):]
                    scale = float(string[7:])
                if 'damp:' in line:
                    # damping
                    string = line[line.index('damp:'):]
                    damp = float(string[6:])
                if 'obj:' in line:
                    # obj
                    string = line[line.index('obj:'):]
                    string = string[string.index('rllab'):-1]
                    obj = '/home/cfinn/code/' + string
                if 'mass:' in line:
                    # mass
                    string = line[line.index('mass:'):]
                    mass = float(string[6:])
                if 'xml:' in line:
                    string = line[line.index('xml:'):]
                    xml_file = string[5:-1]
                    suffix = xml_file[xml_file.index('pusher'):]
                    xml_file = '/home/rosen/rllab_copy/vendor/local_mujoco_models/' + suffix
                if (mass and scale and obj and damp) or xml_file:
                    break
        if not xml_file:
            print(obj)
            print(scale)
            print(mass)
            print(damp)
            model = pusher(mesh_file=obj,mesh_file_path=obj, obj_scale=scale,obj_mass=mass,obj_damping=damp)
            model.save('/home/cfinn/code/gym/gym/envs/mujoco/assets/pusher.xml')
        else:
            copyfile(xml_file, '/home/cfinn/code/gym/gym/envs/mujoco/assets/pusher.xml')
    else:
        # TODO - could call code to autogenerate xml file here
        model = pusher(mesh_file=args.obj_filepath, mesh_file_path=args.obj_filepath)
        model.save('/home/cfinn/code/gym/gym/envs/mujoco/assets/pusher.xml')

    # Copy xml file to gym xml location
    #if args.xml_filepath != 'None':
    #    copyfile(args.xml_filepath, '/home/cfinn/code/gym/gym/envs/mujoco/assets/pusher.xml')
    #else:
    #    model.save('/home/cfinn/code/gym/gym/envs/mujoco/assets/pusher.xml')
    if args.obj_filepath != 'None':
        copy2(args.obj_filepath, '/home/cfinn/code/gym/gym/envs/mujoco/assets')

    env = gym.envs.make('Pusher-v0')
    for _ in range(100000):
        #env.render()
        time.sleep(0.01)
