import pickle
import bz2

class BZ2Pikcle():
    def __init__(self):
        pass
    
    def loads(self,comp):
        return pickle.loads(bz2.decompress(comp))

    def dumps(self,obj):
        return bz2.decompress(pickle.dumps(obj))

    def load(self,fname):
        fin = bz2.BZ2File(fname, 'rb')
        try:
            pkl = fin.read()
        finally:
            fin.close()
        return pickle.loads(pkl)

    def dump(self, obj, fname, level=9):
        pkl = pickle.dumps(obj)
        fout = bz2.BZ2File(fname, 'wb', compresslevel=level)
        try:
            fout.write(pkl)
        finally:
            fout.close()
