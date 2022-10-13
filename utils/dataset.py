import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchvision.transforms import transforms
from PIL import Image
import lmdb
import six
import io
import sys
import random
from typing import List
import numpy as np

"""
lmdb reference: https://lmdb.readthedocs.io/en/release/

class lmdb.Environment(
    path,
    map_size=10485760,
    subdir=True,
    readonly=False,
    metasync=True,
    sync=True,
    map_async=False,
    mode=493,
    create=True,
    readahead=True,
    writemap=False,
    meminit=True,
    max_readers=126,
    max_dbs=0,
    max_spare_txns=1,
    lock=True
)

Structure for a database environment. An environment may contain multiple
databases, all residing in the same shared-memory map and underlying disk file.

To write to the environment a Transaction must be created. One simultaneous
write transaction is allowed, however there is no limit on the number of
read transactions even when a write transaction exists.

Equivalent to mdb_env_open()

# Parameters

 @ path:
    Location of directory (if subdir=True) or file prefix to store the database.
 @ map_size:
    Maximum size database may grow to; used to size the memory mapping.
 If database grows larger than map_size, an exception will be raised and the user
 must close and reopen Environment. On 64-bit there is no penalty for making this
 huge (say 1TB). Must be <2GB on 32-bit.
 @ subdir:
    If True, path refers to a subdirectory to store the data and lock files in,
 otherwise it refers to a filename prefix.
 @ readonly:
    If True, disallow any write operations. Note the lock file is still
 modified. If specified, the write flag to begin() or Transaction is ignored.
 @ metasync:
    If False, flush system buffers to disk only once per transaction,
 omit the metadata flush. Defer that until the system flushes files to disk,
 or next commit or sync().
 This optimization maintains database integrity, but a system crash may undo the last
 committed transaction. I.e. it preserves the ACI (atomicity, consistency, isolation)
 but not D (durability) database property.
 @ sync:
    If False, don’t flush system buffers to disk when committing a transaction.
 This optimization means a system crash can corrupt the database or lose the last
 transactions if buffers are not yet flushed to disk.
 @ mode:
    File creation mode.
 @ create:
    If False, do not create the directory path if it is missing.
 @ readahead:
    If False, LMDB will disable the OS filesystem readahead mechanism,
 which may improve random read performance when a database is larger than RAM.
 @ writemap:
    If True, use a writeable memory map unless readonly=True.
 This is faster and uses fewer mallocs, but loses protection from application bugs like
 wild pointer writes and other bad updates into the database. Incompatible with nested
 transactions.
 @ meminit:
    If False LMDB will not zero-initialize buffers prior to writing them to disk.
 This improves performance but may cause old heap data to be written saved in the unused
 portion of the buffer. Do not use this option if your application manipulates confidential
 data (e.g. plaintext passwords) in memory. This option is only meaningful when writemap=False;
 new pages are always zero-initialized when writemap=True.
 @ map_async:
    When writemap=True, use asynchronous flushes to disk. As with sync=False,
 a system crash can then corrupt the database or lose the last transactions. Calling sync() ensures
 on-disk database integrity until next commit.
 @ max_readers:
    Maximum number of simultaneous read transactions. Can only be set by the first processto open an
 environment, as it affects the size of the lock file and shared memory area. Attempts to simultaneously
 start more than this many read transactions will fail.
 @ max_dbs:
    Maximum number of databases available. If 0, assume environment will be used as a single database.
 @ max_spare_txns: Read-only transactions to cache after becoming unused. Caching transactions avoids
 two allocations, one lock and linear scan of the shared environment per invocation of begin(),
 Transaction, get(), gets(), or cursor(). Should match the process’s maximum expected concurrent
 transactions (e.g. thread count).
 @ lock:
    If False, don’t do any locking. If concurrent access is anticipated, the caller must manage all
 concurrency itself. For proper operation the caller must enforce single-writer semantics, and must
 ensure that no readers are using old transactions while a writer is active. The simplest approach is
 to use an exclusive lock so that no readers may be active at all when a writer begins.


# Method

 % begin(db=None, parent=None, write=False, buffers=False)
    Shortcut for lmdb.Transaction
 % close()
    Close the environment, invalidating any open iterators, cursors, and transactions. Repeat calls
 to close() have no effect.
    Equivalent to mdb_env_close()
 % copy(path, compact=False)
    Make a consistent copy of the environment in the given destination directory. If compact=True,
 perform compaction while copying: omit free pages and sequentially renumber all pages in output.
 This option consumes more CPU and runs more slowly than the default, but may produce a smaller
 output database.
 % copyfd(fd, compact=False)
    Copy a consistent version of the environment to file descriptor fd. If compact=True,
 perform compaction while copying: omit free pages and sequentially renumber all pages in output.
 This option consumes more CPU and runs more slowly than the default, but may produce a smaller
 output database.
 % flags()
    Return a dict describing Environment constructor flags used to instantiate this environment.
 % info()
    Return some nice environment information as a dict:
     $ Keys:
      map_addr:	Address of database map in RAM.
      map_size:	Size of database map in RAM.
      last_pgno: ID of last used page.
      last_txnid: ID of last committed transaction.
      max_readers:	Number of reader slots allocated in the lock file. Equivalent to the value of maxreaders= specified by the first process opening the Environment.
      num_readers:	Maximum number of reader slots in simultaneous use since the lock file was initialized.
 % max_key_size()
    Return the maximum size in bytes of a record’s key part. This matches the MDB_MAXKEYSIZE constant set at compile time.
 % max_readers()
    Return the maximum number of readers specified during open of the environment by the first process. This is the same as max_readers= specified to the constructor
 if this process was the first to open the environment.

"""

class LmdbDataset(Dataset):

    def __init__(
        self,
        root: str,
        task: str = "gen",
        transforms: List[callable] = [],
        target_transforms: List[callable] = [],
        boxes_transforms: List[callable] = []
    ):
        super(LmdbDataset, self).__init__()
        self.env = lmdb.open(
            path = root,
            max_readers = 1,
            readonly = True,
            lock = False,
            readahead = False,
            meminit = False,
            map_size = int(1e9)
        )
        self.task = task

        if not self.env:
            print("creat lmdb from %s failed." % root)
            sys.exit(0)
        with self.env.begin(write = False) as txn:
            nSample_key = "nSamples".encode("utf-8")
            nSamples = int(txn.get(nSample_key))
            self.nSamples = nSamples

        self.transforms = transforms
        self.target_transforms = target_transforms
        self.boxes_transforms = boxes_transforms

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), "ERROR:index out of range"
        with self.env.begin(write = False) as txn:
            img_key = ("image-%09d" % index).encode("utf-8")
            img_buf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(img_buf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert("RGB")
            except IOError:
                print("Corrupted image for %d" % index)
                return self[index + 1]
            # image transfrom
            for t in self.transforms:
                if callable(t):
                    img = t(img)
            

            if self.task == "gen":
                return img

            label_key = ("label-%09d" % index).encode("utf-8")
            label = txn.get(label_key).decode("utf-8")
            
            # label transform
            for t in  self.target_transforms:
                if callable(t):
                    label = self.target_transform(label)
            
            if self.task == "rec":
                return img, label
            
            boxes_key = ("box-%09d" % index).encode("utf-8")
            boxes = txn.get(boxes_key).decode("utf-8")
            for t in  self.boxes_transforms:
                if callable(t):
                    boxes = t(boxes)
            
            return img, label, boxes


class randomSequentialSampler(Sampler):

    def __init__(self, data_source, batch_size):
        # super(randomSequentialSampler, self).__init__()
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        # index = torch.LongTensor(len(self)).fill_(0)
        index = torch.zeros(len(self), dtype = torch.long)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - 1 - self.batch_size)
            # cause the data_source index from not 1 but 0
            batch_index = random_start + torch.arange(self.batch_size)
            index[i * self.batch_size : (i + 1) * self.batch_size] = batch_index

        if tail: # last batch is smaller than batch_size
            random_start = random.randint(0, len(self) - 1 - self.batch_size)
            tail_index = random_start + torch.arange(tail)
            index[(i + 1) * self.batch_size : ] = tail_index

        return iter(index)


class resizeNormalize(object):
    def __init__(
        self,
        size,
        mean = 0.5,
        std = 0.5,
        interpolation = Image.BILINEAR
    ):
        self.size = size # (w, h) instead shape (h,w,c) of array in numpy
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        # self.resize = transforms.Resize(self.size, interpolation = self.interpolation)
        self.toTensor = transforms.ToTensor()

    def __call__(self, img, train = False, normalize = True):
        # img is an instance of PIL.Image
        imgW, imgH = img.size
        new_w, new_h = self.size
        if imgW < new_w:
            imgW = new_w
        size = (imgW, new_h)
        if train:
            size = self.size
        img = img.resize(size, self.interpolation)
        img = self.toTensor(img)
        if normalize:
            img = img.sub_(self.mean).div_(self.std)
        return img

class alignCollate(object):
    
    def __init__(
        self,
        imgH = 64,
        imgW = 64,
        keep_ratio = False,
        min_ratio = 1 # ratio define as imgW / imgH
    ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        imgs, labels, boxes = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for img in imgs:
                w, h = img.size
                ratios.append(w / h)
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)
        transform = resizeNormalize((imgW, imgH))
        imgs = [transform(img, train = True, normalize = True) for img in imgs]
        imgs = torch.cat([t.unsqueeze(0) for t in imgs], 0)
        
        return imgs, labels, boxes


