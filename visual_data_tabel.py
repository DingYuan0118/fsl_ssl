from visualize_datapre import visualize_datapre
from io_utils import parse_args
import os, random

if __name__ == "__main__":

    params = parse_args('mytest')
    loadfile = os.path.join('filelists', params.test_dataset, 'novel.json')
    sub_meta, class_names = visualize_datapre(params, loadfile)
    random.seed(0)

    # params
    test_n_way = params.test_n_way
    test_n_shot = params.test_n_shot
    test_n_query = params.test_n_query
    image_size = params.image_size

    classes_id = sub_meta.keys()
    selected_classes_id = random.sample(classes_id, test_n_way)

    # random selected images_path
    selected_imgs = {}
    for i in selected_classes_id:
        sub_imgs = sub_meta[i]
        sub_selected_imgs = random.sample(sub_imgs, test_n_shot + test_n_query)
        selected_imgs[i] = sub_selected_imgs


    with open("dataset_table.html", "w") as fo:
        fo.write("<html>\n")
        fo.write("<body>\n\n")
        fo.write("<table>\n")
        for key in selected_imgs.keys():
            fo.write("\t<tr>\n")
            fo.write("\t\t<td><font size='5'>{}</font></td>\n".format(class_names[key]))
            for img_path in selected_imgs[key][:test_n_shot]:
                fo.write("\t\t<td><img src={0} height={1} width={1} /></td>\n".format(img_path, image_size))
            fo.write("\t</tr>\n")
        fo.write("</table>\n")
        fo.write("</body>\n")
        fo.write("</html>\n")

         