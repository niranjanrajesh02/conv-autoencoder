from dataset import load_data
from CAE import CAE
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 1

RESULTS_PATH = "/home/niranjan.rajesh_ug23/TNBC/ConvAE"

x, y = load_data()

print(x.shape, y.shape)

model = CAE(input_shape=x.shape[1:], code_dim=10)

optimizer = 'adam'
model.compile(optimizer=optimizer, loss='mse')
hist = model.fit(x, x, batch_size=BATCH_SIZE, epochs=30)

out = model.predict(x)

counter = 1

sns.set(rc={'figure.figsize':(12,25)})
fig =  plt.figure()
rows = 4
for i in range(rows):
  ind = random.randint(0, 1200)
  X_img = Image.fromarray(np.uint8(x[ind] *255)).convert('RGB')
  R_img = Image.fromarray(np.uint8(out[ind] *255)).convert('RGB')
  ax1 = fig.add_subplot(rows,2,counter)
  ax1 = ax1.title.set_text("Input Image")
  ax1 = plt.imshow(X_img)
  counter += 1
  ax2 = fig.add_subplot(rows,2,counter)
  ax2 = ax2.title.set_text("Reconstructed Image")
  ax2 = plt.imshow(R_img)
  counter += 1

fig.savefig(f'{RESULTS_PATH}/Reconstructed_Patches.png')