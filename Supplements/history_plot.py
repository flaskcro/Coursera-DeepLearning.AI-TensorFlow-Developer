import matplotlib.pyplot as plt

def draw_history(history):
    his_dict = history.history
    loss = his_dict['loss']
    val_loss = his_dict['val_loss']
    acc = his_dict['accuracy']
    val_acc = his_dict['val_accuracy']
    
    epochs = range(1, len(loss) + 1)
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs, loss, label='train_loss')
    ax1.plot(epochs, val_loss, label='val_loss')
    ax1.legend()
    ax1.set_title('Loss Graph')
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(epochs, acc, label='train_accuracy')
    ax2.plot(epochs, val_acc, label='val_accuracy')
    ax2.set_title('Accuracy Graph')
    ax2.legend()
    plt.show()