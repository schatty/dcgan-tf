DATADIR=data/flowers

wget -O flowers.tgz http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar zxvf flowers.tgz -C $DATADIR
rm flowers.tgz
mkdir -p $DATADIR
mv $DATADIR/jpg/* $DATADIR/
rmdir $DATADIR/jpg