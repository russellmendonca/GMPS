docker build -t gmps:latest .

docker run --rm -it gmps:latest /bin/bash -c "pushd /root/GMPS; ls; python3 launchers/local_train.py; python3 launchers/local_train.py"
