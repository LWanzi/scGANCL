import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,accuracy_score,silhouette_score
#用于分类的4个指标分别为，AUC,ACC,F-score,ARI
from scipy.optimize import linear_sum_assignment

def get_indicator(function_name,y_label,y_score,data_set_name,embeddings):
    datas = []
    y_true = np.argmax(y_label, axis=1)
    f = open('./compare_funtion/sagan/result/'+str(data_set_name)+'/result_'+str(data_set_name)+'.txt', mode='w')
    for i in range(len(y_score)):
        function_list = []
        y_pred = y_score[i]
        #chi = calinski_harabasz_score(embeddings,y_pred)
        acc_1 = acc(y_true,y_pred)
        SC = silhouette_score(embeddings[i],y_pred,metric='euclidean')
        ari = adjusted_rand_score(y_true,y_pred)
        print("ari:{:.4f}".format(ari))
        nmi = normalized_mutual_info_score(y_true,y_pred)
        print("nmi:{:.4f}".format(nmi))
        function_list.append(SC)
        function_list.append(ari)
        function_list.append(nmi)
        function_list.append(acc_1)
        datas.append(function_list)
        f.writelines("------"+function_name[i]+"------\n")
        f.writelines("SC:"+str(SC)+"\n")
        f.writelines("ari:"+str(ari)+"\n")
        f.writelines("nmi:"+str(nmi)+"\n")
        f.writelines("acc_1:" + str(acc_1)+"\n")
        f.writelines("----------------"+"\n")
    f.close()

    create_multi_bars(function_name,datas,data_set_name)

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def create_multi_bars(labels, datas,data_set_name, tick_step=1, group_gap=0.3, bar_gap=0):
    '''
    labels : x轴坐标标签序列
    datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
    tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''
    # ticks为x轴刻度

    indicater = ["SC","ARI","NMI","ACC"]
    ticks = np.arange(len(indicater)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(labels)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # baseline_x为每组柱子第一个柱子的基准x轴位置，随后的柱子依次递增bar_span即可
    baseline_x = ticks - (group_width - bar_span) / 2
    cmap = plt.cm.Set3
    norm = plt.Normalize(vmin=0, vmax=len(datas))
    for index, y in enumerate(datas):
        plt.bar(baseline_x + index * bar_span, y, bar_width ,label=labels[index],color=cmap(norm(index)))

    plt.ylabel('Scores')
    plt.title(data_set_name)
    # x轴刻度标签位置与x轴刻度一致
    plt.xticks(ticks, indicater)
    plt.legend(labels,fontsize =5,loc='upper right')
    plt.savefig("./compare_funtion/sagan/result/"+str(data_set_name)+"/compare_compare.png")
    plt.show()

if __name__=="__main__":
    y_label = np.array([
        [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 0], [0, 1, 0], [0, 1, 0],
        [0, 0, 1], [0, 0, 1], [0, 0, 1]
    ])
    function_name = ['knn', 'cnn', 'svm']
    y_score = np.array([[
        [0.8, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.1, 0.3],
        [0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.2, 0.75, 0.05],
        [0.05, 0.05, 0.9], [0.1, 0.3, 0.6], [0.12, 0.8, 0.08],
    ], [
        [0.7, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.56, 0.3],
        [0.2, 0.5, 0.3], [0.1, 0.5, 0.3], [0.2, 0.75, 0.12],
        [0.05, 0.05, 0.9], [0.1, 0.8, 0.6], [0.9, 0.8, 0.08],
    ], [
        [0.5, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.7, 0.3],
        [0.2, 0.4, 0.3], [0.1, 0.5, 0.3], [0.2, 0.9, 0.12],
        [0.05, 0.8, 0.9], [0.1, 0.6, 0.6], [0.9, 0.8, 0.08],
    ]])