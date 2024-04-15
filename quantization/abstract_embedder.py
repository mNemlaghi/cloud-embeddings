import numpy as np
import pickle
from hashlib import md5
from pathlib import Path


class AbstractAWSEmbedder:
    type = "classic"

    def __init__(self, boto_client, model_id = "cohere.embed-english-v3", task_name = "default", embeddings_path="precomputed", provider = "cohere", batch_size =90):
        self.model_id= model_id
        self.client = boto_client
        self.task_name=task_name
        self.base_path = Path(embeddings_path)
        self.base_path.mkdir(exist_ok=True)
        self.provider = provider
        self.batch_size = batch_size
    
    def calibrate_embeddings(self):
        ## Abstract that can be used or not.
        pass

    def encode(self, sentences,   **kwargs): #Max 96 embeddings
        self.embedding_path = Path(f"{self.base_path.resolve()}/{self.model_id}-{self.task_name}-{self.hash_chunk(sentences)}.pickle")
        precomputed=self.embedding_path.is_file()
        if precomputed:
            saved = pickle.load(open(self.embedding_path.resolve(), 'rb'))
            res = saved['final_embeddings']
        else:
            self.embedding_path.touch(exist_ok=True)
            chunks = [sentences[i * self.batch_size:(i + 1) * self.batch_size] for i in range((len(sentences) + self.batch_size - 1) // self.batch_size )]
            res = []
            for idx, chunk in enumerate(chunks):
                res.extend(self.batch_encode(chunk))
            dump = {'final_embeddings':res}
            pickle.dump(dump, open(self.embedding_path.resolve(), 'wb'))
        res = [*map(lambda x:x.astype(np.float32), res)]
        self.my_dimension = len(res[0])
        self.byte_size = res[0].itemsize * self.my_dimension
        

        return res

    def create_matrix(self):
        array = []
        pattern = f"{self.model_id}-{self.task_name}-*.pickle"
        for file in self.base_path.glob(pattern):
            saved = pickle.load(open(file.resolve(), 'rb'))
            array.extend(saved['final_embeddings'])     
        X=np.array(array)
        return X

    def get_svd(self):
        self.svd_path = Path(f"{self.base_path.resolve()}/SVD-{self.model_id}-{self.task_name}.pickle")
        svd_computed=self.svd_path.is_file()
        #SVD can take time, so we store it if not found.
        if svd_computed:
            saved = pickle.load(open(self.svd_path.resolve(), 'rb'))
        else:
            U,S,Vh = np.linalg.svd(self.create_matrix(), full_matrices=False)
            saved = {"U":U, "S":S,"Vh":Vh}
            pickle.dump(saved, open(self.svd_path.resolve(), 'wb'))
        return saved
          
    def get_svd_left_rotation(self):
        return self.get_svd()['U']
    
    @staticmethod
    def hash_chunk(sentences):
        ##Just to identify sentences just in case
        first_bit = sentences[0][:10]
        last_bit = sentences[-1][:10]
        bit = first_bit+last_bit
        return md5(bit.encode('utf8')).hexdigest()
    

class BinaryEncoder(AbstractAWSEmbedder):
    type = "binary"
    def turn_binary(self, elt):
        return [int(e) for e in np.binary_repr(elt).zfill(1)]
        
    def binary_quantize_embedding(self, x):
        l = x>0
        packed = np.packbits([self.turn_binary(elt) for elt in l]).astype(np.uint8)
        return packed
    
    def encode(self, sentences, batch_size=90,**kwargs): #Max 96 sentences
        x1 = super().encode(sentences, batch_size=batch_size,**kwargs)
        x2 =  [self.binary_quantize_embedding(y) for y in x1]
        
        l= [list(y) for y in x2]
        self.my_dimension = len(l[0])
        self.byte_size = self.my_dimension* x2[0].itemsize
        return l


class ScalarEncoder(AbstractAWSEmbedder):
    type = "scalar"

    def calibrate_embeddings(self):
        X = self.create_matrix()
        self.Xmin, self.Xmax = X.min(), X.max()
        self.range = (self.Xmax - self.Xmin) / 255
        
    def scalar_quantize(self, x):
        scaled = (x - self.Xmin) / self.range
        return scaled.astype(np.uint8)

    def encode(self, sentences, batch_size=90,**kwargs): #Max 96 sentences
        x1 = super().encode(sentences, batch_size=batch_size,**kwargs)
        x2 =[self.scalar_quantize(y) for y in x1] 
        l = [list(y) for y in x2] ##Turn back as list for commodity
        self.my_dimension = len(l[0])
        self.byte_size = self.my_dimension* x2[0].itemsize

        return l


    
class TernaryEncoder(AbstractAWSEmbedder):
    type = "ternary"

    def calibrate_embeddings(self):
        X = self.create_matrix()
        self.X33, self.X66 = np.quantile(X.flatten(), 1/3), np.quantile(X.flatten(), 2/3)
    
    def turn_ternary(self, elt):
        return [int(e) for e in np.binary_repr(elt).zfill(2)]
        
    def ternary_quantize_embedding(self, x):
        l = [2 if e>self.X66 else (1 if e>self.X33 else 0)  for e in x]
        packed = np.packbits([self.turn_ternary(elt) for elt in l]).astype(np.uint8)
        return packed
    
    def encode(self, sentences, batch_size=90,**kwargs): #Max 96 sentences
        x1 = super().encode(sentences, batch_size=batch_size,**kwargs)
        x2 =  [self.ternary_quantize_embedding(y) for y in x1]
        
        l= [list(y) for y in x2]
        self.my_dimension = len(l[0])
        self.byte_size = self.my_dimension* x2[0].itemsize
        return l

class QuaternaryEncoder(AbstractAWSEmbedder):
    type = "quaternary"

    def calibrate_embeddings(self):
        X = self.create_matrix()
        self.X25, self.X50, self.X75 = np.quantile(X.flatten(), 1/4), np.quantile(X.flatten(), 1/2), np.quantile(X.flatten(), 3/4)
    
    def turn_quaternary(self, elt):
        return [int(e) for e in np.binary_repr(elt).zfill(2)]
        
    def quaternary_quantize_embedding(self, x):
        l = [3 if e>self.X75 else (2 if e>self.X50 else (1 if e>self.X25 else 0))  for e in x]
        packed = np.packbits([self.turn_quaternary(elt) for elt in l]).astype(np.uint8)
        return packed
    
    def encode(self, sentences, batch_size=90,**kwargs): #Max 96 sentences
        x1 = super().encode(sentences, batch_size=batch_size,**kwargs)
        x2 =  [self.quaternary_quantize_embedding(y) for y in x1]
        
        l= [list(y) for y in x2]
        self.my_dimension = len(l[0])
        self.byte_size = self.my_dimension* x2[0].itemsize
        return l


class TernaryRotatedEncoder(TernaryEncoder):
    type = "rotated-ternary"

    def calibrate_embeddings(self):
        X = self.create_matrix()
        U = self.get_svd_left_rotation()
        self.UT = U.T[:,:U.T.shape[0]]
        self.allrotated = self.UT@X.T
        self.X33, self.X66 = np.quantile(self.allrotated.flatten(), 1/3), np.quantile(self.allrotated.flatten(), 2/3)


    def ternary_quantize_embedding(self, x):
        rotated = self.UT@x
        return super().ternary_quantize_embedding(rotated)
        

class BinaryRotatedEncoder(BinaryEncoder):
    type = "rotated-binary"

    def calibrate_embeddings(self):
        X = self.create_matrix()
        U = self.get_svd_left_rotation()
        self.UT = U.T[:,:U.T.shape[0]]

    def binary_quantize_embedding(self, x):
        l = self.UT@x>0
        return super().binary_quantize_embedding(l)
    

class RotatedScalarEncoder(ScalarEncoder):
    type = "rotated-scalar"

    def calibrate_embeddings(self):
        X = self.create_matrix()
        U = self.get_svd_left_rotation()
        self.UT = U.T[:,:U.T.shape[0]]
        self.allrotated = self.UT@X.T
        self.Xmin, self.Xmax = self.allrotated.min(), self.allrotated.max()
        self.range = (self.Xmax - self.Xmin) / 255
    
    def scalar_quantize(self, x):
        x1 = self.UT@x
        return super().scalar_quantize(x1)


class QuaternaryRotatedEncoder(QuaternaryEncoder):
    type = "rotated-quaternary"

    def calibrate_embeddings(self):
        X = self.create_matrix()
        U = self.get_svd_left_rotation()
        self.UT = U.T[:,:U.T.shape[0]]
        self.allrotated = self.UT@X.T
        self.X25, self.X50, self.X75 = np.quantile(self.allrotated.flatten(), 1/4), np.quantile(self.allrotated.flatten(), 1/2), np.quantile(self.allrotated.flatten(), 3/4)


    def ternary_quantize_embedding(self, x):
        rotated = self.UT@x
        return super().quaternary_quantize_embedding(rotated)