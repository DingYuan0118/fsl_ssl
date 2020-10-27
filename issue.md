I found there is a comment in `train.py` as below:

```py
    if params.jigsaw:
        acc, acc_jigsaw = model.test_loop( val_loader)
        writer.add_scalar('val/acc', acc, epoch)
        writer.add_scalar('val/acc_jigsaw', acc_jigsaw, epoch)
    elif params.rotation:
        acc, acc_rotation = model.test_loop( val_loader)
        writer.add_scalar('val/acc', acc, epoch)
        writer.add_scalar('val/acc_rotation', acc_rotation, epoch)
    else:    
        acc = model.test_loop( val_loader)
        writer.add_scalar('val/acc', acc, epoch)
    print("a epoch test process cost{}s".format(time.time() - start_test_time))
    if acc > max_acc :  #for baseline and baseline++, we don't use validation here so we let acc = -1
        print("best model! save...")
        max_acc = acc
        outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
        torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    if ((epoch+1) % params.save_freq==0) or (epoch==stop_epoch-1):
        outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
        torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    print("a epoch cost {} s".format(time.time() - start_epoch_time))
```

this means for each epoch, we do a validation process for model. if I set the `--method baseline`, according to the comment we should get the  acc `-1` returned. But I found that it always returned `0`. Then I found the `test_loop` function for `baseline` as below:

```py
def test_loop(self, val_loader=None):
        if val_loader is not None:
            num_correct = 0
            num_total = 0
            num_correct_jigsaw = 0
            num_total_jigsaw = 0
            for i, inputs in enumerate(val_loader):
                x = inputs[0]
                y = inputs[1]
                if self.jigsaw:
                    loss_proto, loss_jigsaw, acc, acc_jigsaw = self.forward_loss(x, y, inputs[2], inputs[3])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                    num_correct_jigsaw = int(acc_jigsaw*len(inputs[3]))
                    num_total_jigsaw += len(inputs[3].view(-1))
                elif self.rotation:
                    loss_proto, loss_rotation, acc, acc_rotation = self.forward_loss(x, y, inputs[2], inputs[3])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                    num_correct_jigsaw = int(acc_jigsaw*len(inputs[3]))
                    num_total_jigsaw += len(inputs[3].view(-1))
                else:
                    loss, acc = self.forward_loss(x,y)
                num_correct += int(acc*x.shape[0])
                num_total += len(y)
            
            if self.jigsaw or self.rotation:
                return num_correct*100.0/num_total, num_correct_jigsaw*100.0/num_total_jigsaw
            else:
                return num_correct*100.0/num_total

        else:
            if self.jigsaw:
                return -1, -1
            elif self.rotation:
                return -1, -1
            else:
                return -1 #no validation, just save model during iteration
```

It says when `val_loader` is None, the acc will return -1. But in `train.py`, we set the `val_loader` is not None as below:

```py
 if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(image_size, batch_size = params.bs, jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr     = SimpleDataManager(image_size, batch_size = params.bs, jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft)
        val_loader      = val_datamgr.get_data_loader( val_file, aug = False)
```

It means the `test_loop` function will use `self.forward_loss` to compute `loss` and `acc`. 

My question is that why it always return `0`, when `val_loader` is not `None`?

![issue](issue.JPG)
