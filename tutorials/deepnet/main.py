from deepnet.layers import Maxpool
import numpy as np

def main():
    pass
    maxpool = Maxpool(X_dim = (1,4,4), size=2, stride=2)
    I = np.array([1,2,3,4,
                  5,6,7,8,
                  9,10,11,12,
                  13,14,15,16]).reshape(4,4)


    I =np.expand_dims(I,axis=0)#chw
    I =np.expand_dims(I,axis=0)#bchw

    out = maxpool.forward(I)
    grad = maxpool.backward(out)
    print(out)
    print(grad)

if __name__ =="__main__":
    pass
    main()

