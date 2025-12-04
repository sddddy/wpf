import matplotlib.pyplot as plt


def visualize_trends(original_series, trends, title="Trends Visualization"):
    plt.figure(figsize=(10, 6))
    plt.plot(original_series, label="Original Series")
    for i, trend in enumerate(trends):
        plt.plot(trend, label=f"Trend {i + 1}")
    plt.title(title)
    plt.legend()
    plt.show()


def visualize_attention(attn_weights, title="Attention Visualization", idx=0):
    """
    Visualize attention weights for a specific sample in the batch.

    Args:
        attn_weights (torch.Tensor): Attention weights of shape (batch_size, seq_len, seq_len).
        title (str): Title for the plot.
        sample_idx (int): Index of the sample to visualize.
    """
    attn_weights_sample = attn_weights[idx].cpu().detach().numpy()  # 取某一个样本
    plt.figure(figsize=(10, 6))
    plt.imshow(attn_weights_sample, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title(f"{title} (Sample {idx})")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.show()


def adjust_learning_rate(optimizer, epoch, args):
    if args["lradj"] == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()
