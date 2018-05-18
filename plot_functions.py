from timeit import default_timer as timer
import pickle

import matplotlib.pylab as plt
from matplotlib.pyplot import cm



def save_obj(obj, name ):
    with open('data_tests/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data_tests/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def box_plot_final_val(title, final_val_dict, img_file_name):
    label_list = []
    val_list = []
    for k, semi_final_val_dict in final_val_dict.items():
        keys = semi_final_val_dict.keys()
        vals = semi_final_val_dict.values()
        k_lbl = []
        for ke in keys:
            print(ke)
            k_lbl.append(int(ke[-1:])/10)
        label_list.append(k)
        val_list.append(list(vals))

    print(f'label_list: {label_list}, val_list: {val_list}')
    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
    # rectangular box plot
    bplot = plt.boxplot(val_list,
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             labels=label_list)  # will be used to label x-ticks
    plt.title('Dropout type impact')
    plt.ylabel('Validation loss')

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen', 'lightgrey']
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    plt.savefig('../img/' + img_file_name)
    plt.close()

def scatter_plot_final_val(title, final_val_dict, img_file_name):
    color = iter(cm.rainbow(np.linspace(0, 1, len(final_val_dict))))
    for k, semi_final_val_dict in final_val_dict.items():
        keys = semi_final_val_dict.keys()
        vals = semi_final_val_dict.values()
        k_lbl = []
        for ke in keys:
            print(ke)
            k_lbl.append(int(ke[-1:])/10)
        c = next(color)
        plt.plot(k_lbl, vals, c=c, label=k)
    plt.ylabel("val_loss")
    plt.xlabel("dropout scalar")
    plt.legend(loc='upper left')
    plt.savefig('../img/' + img_file_name)
    plt.close()

def plot_all_grad_ep_vals(ep_val_dict, img_file_name):
    plt.xlabel("epoch")
    color=iter(cm.rainbow(np.linspace(0,1,len(ep_val_dict))))
    for k, v in ep_val_dict.items():
        epochs = ep_val_dict[k].keys()
        plt.xticks(np.asarray(list(epochs)))
        val_losses = [item[1] for item in list(ep_val_dict[k].values())]
        c=next(color)
        val_1d = np.gradient(val_losses)
        plt.plot(epochs, val_1d, c=c, label=k)
        plt.ylabel("val_loss_1st_derivative")
        #plt.yscale('log')
    plt.legend(loc='upper left')
    plt.savefig('../img/1vd_'+img_file_name)
    plt.close()


def plot_all_ep_vals(ep_val_dict, img_file_name, perplex=False):

    plt.xlabel("epoch")
    color=iter(cm.rainbow(np.linspace(0,1,len(ep_val_dict))))
    for k, v in ep_val_dict.items():
        epochs = ep_val_dict[k].keys()
        plt.xticks(np.asarray(list(epochs)))
        val_losses = [item[1] for item in list(ep_val_dict[k].values())]
        c=next(color)
        if perplex:
            #convert loss to perplexity
            ey_np = np.exp(np.asarray(val_losses))
            ey_list = ey_np.tolist()
            plt.plot(epochs, ey_list, c=c, label=k)
        else:
            plt.plot(epochs, val_losses, c=c, label=k)
    if perplex:
        plt.ylabel("val_perplexity")
    else:
        plt.ylabel("val_loss")
        plt.yscale('log')
    plt.legend(loc='upper left')
    plt.savefig('../img/'+img_file_name)
    plt.close()

def plot_all_ep_train_val(ep_val_dict, img_file_name):
    plt.ylabel("loss")
    plt.xlabel("epoch")
    color=iter(cm.rainbow(np.linspace(0,1,len(ep_val_dict)*2)))
    for k, v in ep_val_dict.items():
        print(k)
        epochs = ep_val_dict[k].keys()
        plt.xticks(np.asarray(list(epochs)))
        val_losses = [item[1] for item in list(ep_val_dict[k].values())]
        train_losses = [item[0] for item in list(ep_val_dict[k].values())]
        c=next(color)
        plt.plot(epochs, train_losses, c=c, label='train_'+k, ls='dashed')
        c=next(color)
        plt.plot(epochs, val_losses, c=c, label='val_'+k)

    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.savefig('../img/'+img_file_name)
    plt.close()

def plot_all_in_range(prefix, drop_acron, range_strt, range_stop, arch, perplex=False):
    ep_val_dict = {}
    for i in range(range_strt, range_stop):
        pt_txt = str(i/10)
        ep_val_dict[f'{drop_acron}_{pt_txt}'] = load_obj(f'{prefix}_ep_vals_{arch}_{drop_acron}_{i}')
    img_file_name = f'{prefix}_{arch}_{drop_acron}_{range_strt}-{range_stop}.png'
    #plot_all_grad_ep_vals(ep_val_dict, img_file_name)
    plot_all_ep_vals(ep_val_dict, img_file_name, perplex)

def plot_all_in_range_decimal(prefix, drop_acron, range_strt, range_stop, arch, perplex=False):
    ep_val_dict = {}
    for i in range(range_strt, range_stop, 5):
        num_str = str(i/100)
        ep_val_dict[f'{drop_acron}_{num_str}'] = load_obj(f'{prefix}_{drop_acron}_{num_str}')
    img_file_name = f'{prefix}_{arch}_{drop_acron}_{range_strt}-{range_stop}.png'
    #plot_all_grad_ep_vals(ep_val_dict, img_file_name)
    plot_all_ep_vals(ep_val_dict, img_file_name, perplex)

def workflow():
    start = timer()
    data = load_obj('penn_test')
    print(data)
    data = load_obj('penn_drop_wdrop_9')
    print(data)
    end = timer()
    elapsed = end - start
    print(f'>>workflow() took {elapsed}sec')


if __name__ == "__main__":
    workflow()