import urllib.request
import re
import os
import urllib


def get_html(url):
    page = urllib.request.urlopen(url)
    html_a = page.read()
    return html_a.decode('utf-8')


def get_img(html):
    reg = r'https://[^\s]*?\.jpg'
    # Convert to a regular object
    imgre = re.compile(reg)
    # list all the link path and put them into the imgList
    imglist = imgre.findall(html)
    x = 0
    # Setting save path
    path = 'C:\\Users\\JIA\\Pictures\\Page13'

    if not os.path.isdir(path):
        # Create a new path if not exist.
        os.makedirs(path)
    paths = path + '\\'
    for imgurl in imglist:
        # Op0en the imgList, download the image to local.
        urllib.request.urlretrieve(imgurl, '{0}{1}.jpg'.format(paths, x))
        x = x + 1
        print('Download start...')
    return imglist

# Get the detailed information from the web address.
html_b = get_html("https://www.shutterstock.com/zh/search/professor?page=1")
print(get_img(html_b))