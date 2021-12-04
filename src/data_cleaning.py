import os
import math
import operator
import datetime
import multiprocessing as mp
from PIL import Image
from functools import reduce


def read_data_from_page(index):
    num = 0
    imgs = []
    cwd = os.getcwd().replace('\\', '/')
    groups = os.listdir(cwd + '/data')
    for group in groups:
        pages = os.listdir(cwd + '/data/{}'.format(group))
        if num + len(pages) >= index:
            for page in pages:
                num += 1
                if num == index:
                    filenames = os.listdir(cwd + '/data/{}/{}'.format(group, page))
                    for filename in filenames:
                        img = Image.open(cwd + '/data/{}/{}/{}'.format(group, page, filename))
                        imgs.append(img)
                    return imgs
        else:
            num += len(pages)
    return imgs

def same_img(img1, img2, threshold) -> bool:
    h1 = img1.histogram()
    h2 = img2.histogram()
    result = math.sqrt(reduce(operator.add,  list(
        map(lambda a, b: (a-b)**2, h1, h2)))/len(h1))
    return result <= threshold

def process_page(page, threshold):
    imgs = []
    unique_imgs = []
    for img in read_data_from_page(page):
        imgs.append({'data': img, 'repeated': False})
    for i in range(len(imgs)):
        if not imgs[i]['repeated']:
            unique_imgs.append(imgs[i])
            for j in range(i + 1, len(imgs)):
                if same_img(imgs[i]['data'], imgs[j]['data'], threshold):
                    imgs[j]['repeated'] = True
    return unique_imgs

def run(pages, threshold):
    unique_imgs = []
    for page in pages:
        print('[step 1]- {} - now processing page {}'.format(datetime.datetime.now(tz=datetime.timezone.utc).ctime() ,page))
        unique_imgs.extend(process_page(page, threshold))
    return unique_imgs

def generate_output(unique_imgs, dir):
    batch = 0
    count = 0
    cwd = os.getcwd().replace('\\', '/')
    if os.path.exists('{}/data/cleaned/{}'.format(cwd, dir)):
        for batch_dir in os.listdir('{}/data/cleaned/{}'.format(cwd, dir)):
            for filename in os.listdir('{}/data/cleaned/{}/{}'.format(cwd, dir, batch_dir)):
                os.remove('{}/data/cleaned/{}/{}/{}'.format(cwd, dir, batch_dir, filename))
            os.removedirs('{}/data/cleaned/{}/{}/'.format(cwd, dir, batch_dir))
    os.makedirs('{}/data/cleaned/{}/batch_0/'.format(cwd, dir), exist_ok=True)
    for i in range(len(unique_imgs)):
        unique_imgs[i]['data'].save('{}/data/cleaned/{}/batch_{}/{}.jpg'.format(cwd, dir, batch, i))
        count += 1
        if count > 50:
            batch += 1
            count = 0
            os.makedirs('{}/data/cleaned/{}/batch_{}/'.format(cwd, dir, batch), exist_ok=True)

def process_all(imgs):
    print('[step 3]- {} - now processing all images'.format(datetime.datetime.now(tz=datetime.timezone.utc).ctime()))
    unique_imgs = []
    for i in range(len(imgs)):
        if not imgs[i]['repeated']:
            unique_imgs.append(imgs[i])
            for j in range(i + 1, len(imgs)):
                print('[step 3]- {} - ({}, {}) / {}'.format(datetime.datetime.now(tz=datetime.timezone.utc).ctime(), i, j, len(imgs)))
                if same_img(imgs[i]['data'], imgs[j]['data'], threshold):
                    imgs[j]['repeated'] = True
    return unique_imgs

def clean_results():
    cwd = os.getcwd().replace('\\', '/')
    if os.path.exists('{}/data/cleaned'.format(cwd)):
        for batch_dir in os.listdir('{}/data/cleaned'.format(cwd)):
            for filename in os.listdir('{}/data/cleaned/{}'.format(cwd, batch_dir)):
                os.remove('{}/data/cleaned/{}/{}'.format(cwd, batch_dir, filename))
            os.removedirs('{}/data/cleaned/{}/'.format(cwd, batch_dir))
        os.removedirs('{}/data/cleaned')

def get_page_num():
    result = 0
    for group in os.listdir('data/'):
        result += len(os.listdir('data/{}'.format(group)))
    return result

if __name__ == '__main__':
    clean_results()
    ts = datetime.datetime.now()
    num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)

    threshold = 1
    stride = math.ceil(get_page_num()/num_cores)
    pages_on_core = [[j for j in range(i * stride + 1, min((i + 1) * stride + 1, get_page_num()))] for i in range(num_cores)]

    results = [pool.apply_async(run, args=(pages, threshold)) for pages in pages_on_core]
    rough_imgs = []
    for p in results:
        rough_imgs.extend(p.get())

    print('[step 2]- {} - now saving all images in step 1'.format(datetime.datetime.now(tz=datetime.timezone.utc).ctime()))
    generate_output(rough_imgs, 'rough')

    unique_imgs = process_all(rough_imgs)

    print('[step 4]- {} - now saving all images in step 3'.format(datetime.datetime.now(tz=datetime.timezone.utc).ctime()))
    generate_output(unique_imgs, 'unique')

    te = datetime.datetime.now()
    print('[step 5] - {} - total time elapsed: {}s'.format(datetime.datetime.now(tz=datetime.timezone.utc).ctime(), (te - ts).total_seconds()))