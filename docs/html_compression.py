import re
import time
import htmlmin


def replaceHtmlResource(htmlCode, tag, timeStamp):
    return htmlCode.replace('.{}\"'.format(tag), '.{}?_t={}\"'.format(tag, timeStamp))


if __name__ == '__main__':
    with open('index_write.html', 'r') as f:
        htmlCode = f.read()
    timeStamp = int(time.time())
    timeStr = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
    htmlCode = replaceHtmlResource(htmlCode, 'jpg', timeStamp)
    htmlCode = replaceHtmlResource(htmlCode, 'png', timeStamp)
    htmlCode = replaceHtmlResource(htmlCode, 'svg', timeStamp)

    htmlCode = htmlmin.minify(htmlCode, remove_comments=True, remove_empty_space=True)
    with open('./index.html', 'w') as f:
        f.write(htmlCode)