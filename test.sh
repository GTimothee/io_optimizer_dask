#dd if=/dev/zero of='/run/media/user/HDD 1TB/random.txt' bs=1M count=1024 conv=fdatasync,notrunc status=progress
echo 3 | sudo tee /proc/sys/vm/drop_caches
echo 'Read:'
dd if='/run/media/user/HDD 1TB/data/randomfile.npy' of=/dev/null bs=1M count=1024 status=progress
