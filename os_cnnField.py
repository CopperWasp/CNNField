import os_cnn as cnn
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable


class cnnField:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.classifiers = {}  # class: (model, optimizer)
        self.correct_counter = 0
        self.false_counter = 0  # not the best way to keep track of accuracy etc.
        self.yolo_counter = 0

    def extendField(self, new_obj):
        net = cnn.Net()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        self.classifiers[new_obj] = (net, optimizer)

    def updateCNN(self, key, input, label):
        (clf, opt) = self.classifiers[key]
        opt.zero_grad()
        output = clf(input)
        loss = self.criterion(output, label)
        loss.backward()
        opt.step()
        self.classifiers[key] = (clf, opt) # save back
        return output # return prediction if needed

    def fieldPredict(self, input):
        outputs = {}
        for key in self.classifiers.keys():
            clf = self.classifiers[key][0]
            out = clf(input)
            outputs[key] = torch.argmax(out)
        return outputs

    def updateField(self, input, supervision):
        field_output = self.fieldPredict(input)
        self.yolo_counter += len(supervision.keys())  # keep track of num. objs. yolo detected
        # supervision e.g. {b'car': 0.6168470978736877, b'person': 0.7184239625930786}
        intersect = list(set(supervision.keys()).intersection(set(self.classifiers.keys())))
        field_only = list(set(self.classifiers.keys()) - set(supervision.keys()))
        supervision_only = list(set(supervision.keys() - self.classifiers.keys()))

        # if field does not have the object(s) that supervision has (yet), extend field:
        for obj in supervision_only:
            self.extendField(obj)

        # if supervision does not have object(s) that field returned positive, then those field components misclassified
        for obj in field_only:
            if field_output[obj].numpy() == 1:  # output is 1
                self.updateCNN(obj, input, Variable(torch.tensor(0)).unsqueeze(0))
                self.false_counter += 1  # false negative

        # if field agrees with supervision -> correct classification
        for obj in intersect:
            if field_output[obj].numpy() == 1:
                self.updateCNN(obj, input, Variable(torch.tensor(1)).unsqueeze(0))
                self.correct_counter += 1
            elif field_output[obj].numpy() == 0:
                self.updateCNN(obj, input, Variable(torch.tensor(0)).unsqueeze(0))
                self.false_counter += 1  # false positive

        print("\tYolo:")
        print("\t"+str(supervision))
        print("\tField:")
        print("\t"+str(field_output))
        print("\tCorrect, False, Yolo:")
        print("\t"+str(self.correct_counter)+", "+str(self.false_counter)+", " +str(self.yolo_counter))










