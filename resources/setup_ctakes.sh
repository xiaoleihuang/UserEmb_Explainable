wget https://mirrors.gigenet.com/apache//ctakes/ctakes-4.0.0.1/apache-ctakes-4.0.0.1-bin.tar.gz
tar -zxvf apache-ctakes-4.0.0.1-bin.tar.gz
rm apache-ctakes-4.0.0.1-bin.tar.gz
wget https://cfhcable.dl.sourceforge.net/project/ctakesresources/ctakes-resources-4.0-bin.zip
unzip ctakes-resources-4.0-bin.zip
cp -r ./resources/* /usr/local/apache-ctakes-4.0.0/resources/
rm -rf ./resources/ ctakes-resources-4.0-bin.zip

