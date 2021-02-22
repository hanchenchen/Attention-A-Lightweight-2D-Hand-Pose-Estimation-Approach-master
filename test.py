import torch

if torch.cuda.is_available():

    # creates a LongTensor and transfers it
    # to GPU as torch.cuda.LongTensor
    a = torch.full((10,), 3, device=torch.device("cuda"))
    print(type(a))
    b = a.to(torch.device("cpu"))
    # transfers it to CPU, back to
    # being a torch.LongTensor
'''print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
model = create_model()
name = 'test'
filepath = 'test\\' + name + '.hdf5'
model.load_weights(filepath)

f = open("test\\log.bak", 'w+')
img_path = 'test\\dataset\\rgb\\00000119.jpg'
print(model(open(img_path, 'r')))'''



