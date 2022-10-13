import lmdb
import cv2
import numpy as np
import os
import glob




def bytes2img(img_bytes):
    imgBuf = np.frombuffer(img_bytes, dtype = np.uint8)
    img = cv2.imdecode(imgBuf, cv2.IMREAD_UNCHANGED)
    return img

def read_img_as_bytes(img_path: str) -> bytes:
    with open(img_path, "rb") as rf:
        img_bytes = rf.read()
    return img_bytes


def img2bytes(img):
    'convert img to bytes string'
    imgbuf = cv2.imencode('.jpg', img)[1]
    imgBin = imgbuf.tobytes()
    return imgBin

def checkImageIsValid(img_bytes):
    '''
    Checking if an image is valid
    # Input
     @ imgBin: an instance of bytes converted from numpy.ndarray
    # Output: bool
    '''
    if img_bytes is None:
        return False
    img = bytes2img(img_bytes)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    '''
    write data to lmdb
    # Input
     @env: a lmdb.Environment instance, the container of database
     @cache: a dict instance, which cache the data of imgs and annotations
    '''
    # Transaction commits automatically, txn is an instance of lmdb.Transaction
    with env.begin(write = True) as txn: 
        for k, v in cache.items():
            txn.put(k, v)

    
        

def create_tiny_imageNet_DataSet(out_path: str, origin_src: str, check_valid: bool = True, map_size: int = np.uint(1e12)):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    env = lmdb.open(out_path, map_size = map_size)
    cache = dict()
    cnt = 0
    annotations = glob.iglob(f"{origin_src}/*/*.txt")
    for i, anno in enumerate(annotations):
        dir_name = os.path.dirname(anno)
        imgs_dir = f"{dir_name}/images"
        class_id = dir_name.split("/")[-1]
        with open(anno) as rf:
            for line in iter(rf.readline, ""):
                try:
                    segs = line.split()
                    
                    img_name, boxes = segs[0], " ".join(segs[1:])
                    img_path = f"{imgs_dir}/{img_name}"
                    img_bytes = read_img_as_bytes(img_path)
                    
                    if check_valid:
                        if not checkImageIsValid(img_bytes):
                            print('%s is not a valid image' % img_name)
                            continue
                    
                    imageKey = ("image-%09d" % cnt).encode('utf-8')
                    labelKey = ("label-%09d" % cnt).encode('utf-8')
                    boxKey = ("box-%09d" % cnt).encode("utf-8")
                    cache[imageKey] = img_bytes
                    cache[labelKey] = class_id.encode("utf-8")
                    cache[boxKey] = boxes.encode("utf-8")
                    cnt += 1

                except Exception as e:
                    print(f"fetch Error: {e}")
                    continue

                
                if cnt % 1000 == 0:
                    writeCache(env, cache)
                    del cache
                    cache = dict()
                    print(f"{cnt} samples cached")
                
    nSamples = cnt
    cache["nSamples".encode("utf-8")] = str(nSamples).encode("utf-8")
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

def create_tiny_imageNet_val_DataSet(out_path: str, origin_src: str, check_valid: bool = True, map_size: int = np.uint(1e12)):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    env = lmdb.open(out_path, map_size = map_size)
    cache = dict()
    cnt = 0
    annotations = glob.iglob(f"{origin_src}/*.txt")
    for i, anno in enumerate(annotations):
        dir_name = os.path.dirname(anno)
        imgs_dir = f"{dir_name}/images"
        with open(anno) as rf:
            for line in iter(rf.readline, ""):
                try:
                    segs = line.split()
                    
                    img_name, class_id, boxes = segs[0], segs[1], " ".join(segs[2:])
                    img_path = f"{imgs_dir}/{img_name}"
                    img_bytes = read_img_as_bytes(img_path)
                    
                    if check_valid:
                        if not checkImageIsValid(img_bytes):
                            print('%s is not a valid image' % img_name)
                            continue
                    
                    imageKey = ("image-%09d" % cnt).encode('utf-8')
                    labelKey = ("label-%09d" % cnt).encode('utf-8')
                    boxKey = ("box-%09d" % cnt).encode("utf-8")
                    cache[imageKey] = img_bytes
                    cache[labelKey] = class_id.encode("utf-8")
                    cache[boxKey] = boxes.encode("utf-8")
                    cnt += 1

                except Exception as e:
                    print(f"fetch Error: {e}")
                    continue
                
                if cnt % 1000 == 0:
                    writeCache(env, cache)
                    del cache
                    cache = dict()
                    print(f"{cnt} samples cached")
                
    nSamples = cnt
    cache["nSamples".encode("utf-8")] = str(nSamples).encode("utf-8")
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)
        

def create_celeba_aligned_dataset(out_path: str, origin_src: str, check_valid: bool = True, map_size: int = np.uint(1e12)):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    env = lmdb.open(out_path, map_size = map_size)
    cache = dict()
    cnt = 0
    for img_name in glob.iglob(f"{origin_src}/*.jpg"):
        try:
            img_bytes = read_img_as_bytes(img_name)
            if check_valid:
                if not checkImageIsValid(img_bytes):
                    print('%s is not a valid image' % img_name)
                    continue
            
        except Exception as e:
            print(str(e))
            continue

        imageKey = ("image-%09d" % cnt).encode('utf-8')
        cache[imageKey] = img_bytes
        cnt += 1

        if cnt % 1000 == 0:
            writeCache(env, cache)
            del cache
            cache = dict()
            print(f"{cnt} samples cached")

    nSamples = cnt
    cache["nSamples".encode("utf-8")] = str(nSamples).encode("utf-8")
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

# create_tiny_imageNet_DataSet("data/tiny_lmdb/train", "tiny-imagenet-200/train")
 
# create_tiny_imageNet_val_DataSet("data/tiny_lmdb/val", "tiny-imagenet-200/val")


create_celeba_aligned_dataset("data/celeba", "celeba")