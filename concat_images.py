from PIL import Image
import os
import tqdm

dets_img_path1 = "/home/liujianwei/project/dataset/zmot/0804/centerpoint/2D/"
dets_img_path2 = "/home/liujianwei/project/dataset/zmot/0804/mono3d/2D/"
bev_img_path1 = "/home/liujianwei/project/dataset/zmot/0804/centerpoint/BEV/"
bev_img_path2 = "/home/liujianwei/project/dataset/zmot/0804/mono3d/BEV/"

save_path = "/home/liujianwei/project/dataset/zmot/0804/compare2/"
os.makedirs(save_path, exist_ok=True)

img_num = 2536

COL = 3 #指定拼接图片的列数
ROW = 2 #指定拼接图片的行数
UNIT_HEIGHT_SIZE = 1200 #图片高度
UNIT_WIDTH_SIZE = 800 #图片宽度
SAVE_QUALITY = 80 #保存的图片的质量 可选0-100

#进行图片的复制拼接
def concat_images(image_names, name, path):
    inter = 10
    image_files = []
    for img_name in image_names:
        image_files.append(Image.open(img_name)) #读取所有用于拼接的图片
    h = image_files[0].height
    w = image_files[0].width
    w *= 0.4
    w = int(w)
    h *= 0.4
    h = int (h)
    H = max(h * 2 + inter * 3, image_files[2].height)
    W = w + image_files[2].width * 2 + inter * 2
    target = Image.new('RGB', (W, H)) #创建成品图的画布
    #第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
    # image_files[0].resize((1200, 756), Image.BILINEAR).transpose(Image.ROTATE_90).save('./tmp.png')
    target.paste(image_files[0].resize((w, h)), (0, 0, w, h))
    target.paste(image_files[1].resize((w, h)), (0, h + inter * 3, w, h * 2 + inter * 3))
    target.paste(image_files[2], (w + inter, 0, image_files[2].width + w + inter, image_files[2].height))
    target.paste(image_files[3], (w + inter * 2 + image_files[2].width, 0, W, image_files[3].height))
    target.save(path + name + '.png', quality=SAVE_QUALITY) #成品图保存

#获取需要拼接图片的名称
def main():
    for i in tqdm.tqdm(range(1, img_num)):
        det_img_name1 = dets_img_path1 + str(i) + '.png'
        det_img_name2 = dets_img_path2 + str(i) + '.png'
        bev_img_name1 = bev_img_path1 + str(i) + '.png'
        bev_img_name2 = bev_img_path2 + str(i) + '.png'
        image_names = [det_img_name1, det_img_name2, bev_img_name1, bev_img_name2]
        concat_images(image_names, str(i), save_path)

if __name__ == '__main__':
    main()