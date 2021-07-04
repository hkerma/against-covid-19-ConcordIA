import torch
import tqdm
import os
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np 

# from toolsp.visualization import VisdomLinePlotter

## Uncomment for Visdom
# global plotter
# plotter = VisdomLinePlotter(env_name='Training')

def train(model, train_loader, test_loader, device, optimizer, criterion, scheduler, batch_size, num_epochs=25, competition_loader=None):

    best_acc = 0
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../saved_models/best_model.pt")

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        running_corrects = 0.0

        with tqdm.tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:

                tepoch.set_description(f"Epoch {epoch}")

                images.to(device)
                labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                corrects = torch.sum(preds == labels.data)
                running_loss += loss.item()*images.size(0)
                running_corrects += corrects.item()

                accuracy = corrects.cpu().numpy() / batch_size

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

        scheduler.step()

        epoch_loss = running_loss / train_loader.dataset.__len__()
        epoch_acc = running_corrects / train_loader.dataset.__len__()

        ## Uncomment for Visdom
        # plotter.plot('loss', 'train', 'Loss', epoch, epoch_loss)
        # plotter.plot('acc', 'train', 'Accuracy', epoch, epoch_acc*100)


        print('Training summary ------------------------- ')
        print('Epoch {}/{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, num_epochs, epoch_loss, epoch_acc))
        print('------------------------------------------ ')

        test_acc = test(model, test_loader, device, optimizer, criterion, epoch)
        results = competition(model, competition_loader, device, optimizer)

        percent = 0
        for r in results:
            if r == 1:
                percent += 1

        print('Competition summary ---------------------- ')
        percent = (float(percent)/len(results))*100
        print('positive:', percent)
        print('------------------------------------------ ')

        # plotter.plot('competition', 'percent', 'Competition', epoch, percent)

        print()

        if (best_acc < test_acc):
            best_acc = test_acc
            filename = str(percent) + "_" + str(np.around(test_acc.cpu().numpy(), 3))

            sub_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../submissions/' + filename + '.txt')
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../saved_models/' + filename + '.pt') 

            if os.path.exists(sub_path):
                os.remove(sub_path)

            with open(sub_path, 'a') as f:
                for r in results:
                    f.write(str(r) + '\n')

            print('Saving model {} ---------------------- '.format(filename+".pt"))
            torch.save(model.state_dict(), model_path)
            
        print()



def test(model, test_loader, device, optimizer, criterion, epoch):

    model.eval()
    running_loss = 0.0
    running_corrects = 0.0

    all_preds = []
    all_labels = []

    for images, labels in tqdm.tqdm(test_loader, desc='Testing'):

            images.to(device)
            labels.to(device)

            all_labels += labels.data.cpu().numpy().tolist()


            optimizer.zero_grad()
            with torch.no_grad():

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

            all_preds += preds.data.cpu().numpy().tolist()

            running_loss += loss.item()*images.size(0)
            running_corrects += torch.sum(preds == labels.data)



    epoch_loss = running_loss / test_loader.dataset.__len__()
    epoch_acc = running_corrects / test_loader.dataset.__len__()

    # plotter.plot('loss', 'val', 'Loss', epoch, epoch_loss)
    # plotter.plot('acc', 'val', 'Accuracy', epoch, epoch_acc.item()*100)

    print('Val summary ----------------------------- ')
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    precision, recall, _, _= score(np.array(all_labels), np.array(all_preds), average=None, labels=[0, 1])
    sp, sn, pp, pn = np.around(recall[1], 2), np.around(recall[0], 2), np.around(precision[1], 2), np.around(precision[0], 2)

    print('SP:', sp, 'SN:', sn, 'PP:', pp, 'PN:', pn) 

    print('Score:', 6*sp + 5*sn + 3*pp + 2*pn)

    # plotter.plot('competition', 'score', 'Competition', epoch,  6*sp + 5*sn + 3*pp + 2*pn)

    print('------------------------------------------ ')

    return(epoch_acc)


def competition(model, competition_loader, device, optimizer):

    model.eval()

    all_preds = []

    for images in tqdm.tqdm(competition_loader, desc='Competition'):
            images.to(device)

            optimizer.zero_grad()
            with torch.no_grad():

                outputs = model(images)
                #_, preds = torch.max(outputs, 1)

            all_preds += outputs

    return(all_preds)