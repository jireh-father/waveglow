import torch
from distributed import reduce_tensor
import datetime
import time


def eval(eval_loader, model, criterion, num_gpus, start_time, epoch):
    print("[%s] start evaluation" % datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    with torch.no_grad():
        model.eval()
        total_loss = 0.
        for i, batch in enumerate(eval_loader):
            model.zero_grad()

            mel, audio = batch
            mel = torch.autograd.Variable(mel.cuda())
            audio = torch.autograd.Variable(audio.cuda())

            outputs = model((mel, audio))

            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            total_loss += reduced_loss
            if i > 0 and i % 100:
                print("[{}][els: {}] {}/{}:\t{:.9f}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                                         time.time() - start_time, i, len(eval_loader), reduced_loss))
    print("[{}][els: {}] {} epoch :\tavg loss {:.9f}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                                               time.time() - start_time, epoch,
                                                               total_loss / len(eval_loader)))