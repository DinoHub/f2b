# Checks if gdown is installed
if ! type "gdown" > /dev/null; then
    read -p "Install gdown (google drive downloader) with pip3 (y/n)? " yn
    case $yn in
        [Yy]* ) pip3 install gdown;;
        * ) echo "Not installing gdown. Exiting.."; exit;;
    esac
fi

# Downloads with gdown
gdown -O cvat_annot_coco.json https://drive.google.com/uc?id=1KU8qH39kdZ8PLdRMvU0QkIQCEskm8HSH