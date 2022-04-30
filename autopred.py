import os, ljqpy, random, sys, shutil
import torch
import cv2

startPt = None
def mouse_func(event, x, y, flags, param):
    global startPt, img, rects
    yy, xx = img.shape[:2]
    px, py = x/xx, y/yy
    if event == cv2.EVENT_LBUTTONDOWN:
        startPt = px, py
        print('down', px, py)
    if event == cv2.EVENT_LBUTTONUP:
        print('up', startPt)
        if startPt is not None:
            xs, ys = startPt
            xe, ye = px, py
            xmin, xmax = min(xs, xe), max(xs, xe)
            ymin, ymax = min(ys, ye), max(ys, ye)
            xcen, ycen = (xmin+xmax)*0.5, (ymin+ymax)*0.5
            ww, hh = xmax-xmin, ymax-ymin
            rects.append([65, xcen, ycen, ww, hh])
            print('-'*10)
            print(f'{len(rects)} rects')
            for r in rects: print(r)
            ReDraw()
            startPt = None
    if event == cv2.EVENT_MOUSEMOVE:
        if startPt is not None:
            xs, ys = startPt
            xe, ye = px, py
            xmin, xmax = min(xs, xe), max(xs, xe)
            ymin, ymax = min(ys, ye), max(ys, ye)
            xcen, ycen = (xmin+xmax)*0.5, (ymin+ymax)*0.5
            ww, hh = xmax-xmin, ymax-ymin
            ReDraw( (xcen, ycen, ww, hh) )
    if event == cv2.EVENT_RBUTTONDOWN:
        marks = [True for r in rects]
        for i, r in enumerate(rects):
            xcen, ycen, ww, hh = r[1:]
            xmin, xmax = (xcen-ww*0.5), (xcen+ww*0.5)
            ymin, ymax = (ycen-hh*0.5), (ycen+hh*0.5)
            if xmin < px < xmax and ymin < py < ymax:
                marks[i] = False
        rects = [r for i, r in enumerate(rects) if marks[i]]
        ReDraw()

def ReDraw(tempRect=None):
    cpimg = img.copy()
    for rr in rects:
        drawrect(cpimg, rr[1], rr[2], rr[3], rr[4])
    if tempRect:
        xcen, ycen, ww, hh = tempRect
        drawrect(cpimg, xcen, ycen, ww, hh, color=(255,0,0))
    cv2.imshow('a', cpimg)
    
cv2.namedWindow("a")
cv2.setMouseCallback("a", mouse_func)

model = torch.hub.load('yolov5-master', 'custom', path='yolov5s.pt', source='local')
#model = torch.hub.load('yolov5-master', 'custom', path='best.pt', source='local')
model.conf = 0.1
model.iou = 0.1

datadir = 'unlabel'
for fn in ljqpy.ListDirFiles(datadir):
    nfn = f'{random.randint(0, 9999999999)}.png'
    os.rename(fn, os.path.join(datadir, nfn))

def drawrect(img, xcen, ycen, ww, hh, color=(0,255,0)):
    yy, xx = img.shape[:2]
    xmin, xmax = int((xcen-ww*0.5)*xx), int((xcen+ww*0.5)*xx)
    ymin, ymax = int((ycen-hh*0.5)*yy), int((ycen+hh*0.5)*yy)
    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color=color, thickness=5)


for fn in os.listdir(datadir):
    ffn = os.path.join(datadir, fn)
    results = model(ffn)  
    rr = results.pandas().xyxy[0]
    mat = results.pandas().xyxy[0].values   

    img = cv2.imread(ffn)
    yy, xx = img.shape[:2]
    rects = []
    print('-'*30)
    for mm in mat:
        if mm[-1] != 'remote': continue
        xmin, ymin, xmax, ymax = mm[:4]
        xcen, ycen = (xmin+xmax)*0.5/xx, (ymin+ymax)*0.5/yy
        ww, hh = (xmax-xmin)/xx, (ymax-ymin)/yy
        rects.append([mm[5], xcen, ycen, ww, hh])
        print(mm[-1])

    ff = 800 / img.shape[0]
    img = cv2.resize(img, dsize=None, fx=ff, fy=ff)

    for rr in rects: print(rr)
        
    ReDraw()
    kkey = cv2.waitKey(0)

    if kkey == ord('y'):
        print('confirmed')
        with open(os.path.join(datadir, '../datasets/cvoid/labels/train/', fn.replace('.png', '.txt')), 'w') as fout:
            for rr in rects:
                rvs = ['65'] + [f'{x:.4f}' for x in rr[1:]]
                fout.write(' '.join(rvs)+'\n')
            shutil.move(ffn, os.path.join(datadir, '../datasets/cvoid/images/train/', fn))

    #results.show()
