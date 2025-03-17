import pickle
def get_loss(pickle_path):
    # 打开并加载 loss.pickle 文件
    with open(pickle_path, 'rb') as f:
        loss_data = pickle.load(f)
        print(loss_data[0][2])
        pre_epoch = 0
        loss_sum = 0
        pre_loss = []
        loss_count = 0
        for loss in loss_data:
            if loss[0] == pre_epoch:
                pre_loss = loss
                loss_sum+=loss[-1]
                loss_count+=1
            else:
                print(loss_sum/loss_count)
                loss_count = 0
                loss_sum = 0
                pre_epoch = loss[0]
    print(loss_sum/loss_count)
    # 打印加载的数据


if __name__ == "__main__":
    pickle_path = "D:\projects\llh-code-prediction-transformer\output\\travtrans\losses.pickle"
    get_loss(pickle_path)