import torch
import torch.utils.data as du
from tqdm import tqdm


def GRU_training(model,
                 train_dataloader,
                 test_dataloader,
                 cnf,
                 logger=None):
    assert (isinstance(train_dataloader, du.DataLoader))
    assert (isinstance(test_dataloader, du.DataLoader))

    loss_fn = torch.nn.MSELoss(reduction='mean')
    hist = []
    hist_test = []
    optimizer_cnf = cnf['optimizer']
    Optimizer = getattr(torch.optim, optimizer_cnf['class'])
    optimiser = Optimizer(model.parameters(), lr=cnf['learning_rate'], **optimizer_cnf['args'])

    for t in tqdm(range(cnf['num_epochs'])):
        batch_losses = 0
        len_train = 0
        model.train()
        for batch_idx, (x, y) in enumerate(train_dataloader):
            model.zero_grad()
            # Forward pass
            y_pred = model(x)
            loss = torch.sqrt(loss_fn(y_pred, y))
            train_res = float(loss.item())
            print(train_res)
            # Backward pass
            loss.backward()

            optimiser.step()
            optimiser.zero_grad()

            batch_losses += float(train_res)
            len_train += 1
        hist.append(batch_losses / len_train)

        # Test loss
        model.eval()
        test_batch_losses = 0
        len_test = 0
        for test_idx, (x_test, y_test) in enumerate(test_dataloader):
            model.zero_grad()
            with torch.no_grad():
                y_test_pred = model(x_test)
                test_res = float(torch.sqrt(loss_fn(y_test_pred, y_test)).item())
                test_batch_losses += test_res
                len_test += 1
        test_loss = test_batch_losses / len_test
        hist_test.append(test_loss)

    print("Total num epochs trained:", t)
    print("Final training loss:", hist[-1])
    print("Final testing loss:", hist_test[-1])

    res_dict = {}
    res_dict['final_training_loss'] = hist[-1]
    res_dict['final_testing_loss'] = hist_test[-1]
    res_dict['epochs_trained'] = t

    if logger:
        logger.dump_results(model, hist_train=hist, hist_test=hist_test, res_dict=res_dict)

    # Return testing loss for the BO loop
    return hist_test[-1]
