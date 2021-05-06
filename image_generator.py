import matplotlib.pyplot as plt


# since I forgot to save one of the images (accuracy per epoch)...

def plot_accuracy():
    plt.plot([.867, .874, .884, .890, .883, .896, .875, .890, .896, .904, .905, .905, .896, .905, .905], label='train_accuracy', marker='*')
    plt.plot([.870, .881, .893, .872, .895, .897, .887, .881, .903, .905, .905, .895, .904, .905, .905], label='val_accuracy',  marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('first_train/accuracy.png')
    plt.show()

plot_accuracy()