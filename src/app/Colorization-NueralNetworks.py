import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
'''   
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])
'''
np.random.seed(1)

def ANN(X,y,syn0,syn1):

    assert(X.shape[0] == y.shape[0])
    assert(X.shape[1] == syn0.shape[0])
    assert(syn1.shape[1] == y.shape[1])
    for j in range(X.shape[0]):

        # Feed forward through layers 0, 1, and 2
        l0 = X[j,:]
        l1 = nonlin(np.dot(l0,syn0))

        l2 = nonlin(np.dot(l1,syn1))
        l2_error = y[j,:] - l2
        
    
        if (j% 10000) == 0:
            print ("Error:" + str(np.mean(np.abs(l2_error))))
        
        l2_delta = l2_error*nonlin(l2,deriv=True)
		l1_error = l2_delta.dot(syn1.T)
    
        l1_delta = l1_error * nonlin(l1,deriv=True)

        l1 = l1.reshape((5,1))
        l2_delta = l2_delta.reshape((1,3))
        l0 = l0.reshape((9,1))
        l1_delta = l1_delta.reshape((1,5))
        syn1 += l1.dot(l2_delta)
        syn0 += l0.dot(l1_delta)
    return (syn0,syn1)

def test_ANN(X,syn0,syn1):
    y = np.zeros((X.shape[0],3))
    for j in range(X.shape[0]):
        l0 = X[j,:].reshape((1,9))
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        y[j,:] = l2
    return y



from PIL import Image
import numpy as np
import csv

def fetchFromCSV(input_csv):
    input_data = []
    with open(input_csv) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            input_data.append(row)
    return np.array(input_data)

def getArrayOfShape(input_data,h,w,type):
    if(type == 'color'):
        return np.array(input_data).reshape(h,w,3).astype(np.uint8)
    elif(type == 'test'):
        return np.array(input_data).reshape(h,w).astype(np.uint8)
    else:
        return np.array(input_data).reshape(h,w,9).astype(np.uint8)

def showImage(img, flag):
    img = Image.fromarray(img, flag)
    img.show()
    
sample_op_csv = 'color.csv'
sample_ip_csv = 'input.csv'

test_ip_data = 'data.csv'

input_data = fetchFromCSV(sample_ip_csv)
output_data = fetchFromCSV(sample_op_csv)
test_ip_data = fetchFromCSV(test_ip_data)

w, h = 281, 174
op_image = getArrayOfShape(output_data,h,w,'color')
ip_image = getArrayOfShape(input_data,h,w,'bw')
print(input_data.shape, output_data.shape, test_ip_data.shape)

syn0 = 2*np.random.random((9,5)) - 1
syn1 = 2*np.random.random((5,3)) - 1
'''
syn0 = -255*np.random.random((9,5)) + 255
syn1 = -255*np.random.random((5,3)) + 255
'''

print(syn0,syn1)
op = output_data.astype(np.float64)
ip = input_data.astype(np.float64)
print(ip.dtype,op.dtype)
syn0,syn1 = ANN(ip,op, syn0, syn1)

l2 = test_ANN(ip.astype(np.float64),syn0.astype(np.float64),syn1.astype(np.float64))
# l2 = test_ANN(test_ip_data.astype(np.float64),syn0.astype(np.float64),syn1.astype(np.float64))
print(l2)
print(l2.shape)

w1, h1 = 641, 361
# print(np.array(test_ip_data).shape)
# test_inp_image = getArrayOfShape(test_ip_data,h1,w1,'bw')

showImage(l2,'RGB')

# showImage(test_inp_image, 'L')
'''
print(ip_image.shape)
new_ip_img = np.zeros((h,w))
print(new_ip_img[5,7])
for i in range(h):
    for j in range(w):
        new_ip_img[i,j] = ip_image[i,j,4]
print(new_ip_img.shape)

showImage(new_ip_img, 'L')
'''
