
# import numpy as np

# def next_batch(num, data, labels):
#     index = np.arange(0 , len(data))
#     np.random.shuffle(index)
#     index = index[:num]
#     data_shuffle = [data[i] for i in index]
#     labels_shuffle = [labels[i] for i in index]

#     return np.asarray(data_shuffle), np.asarray(labels_shuffle)

class Data():
    def __init__(self,samples,labels):
        self.samples = samples
        self.labels = labels
        self.pointer = 0
    def next_batch(self,num):
        if self.pointer+num <= len(self.samples):
            batch_samples = self.samples[self.pointer:self.pointer+num]
            batch_labels = self.labels[self.pointer:self.pointer+num]
            self.pointer += num
            return batch_samples, batch_labels
        else:
            new_pointer = self.pointer+num - len(self.samples)
            batch_samples = self.samples[self.pointer:-1] + self.samples[:new_pointer]
            batch_labels = self.labels[self.pointer:-1] + self.labels[:new_pointer]
            self.pointer = new_pointer
            return batch_samples, batch_labels