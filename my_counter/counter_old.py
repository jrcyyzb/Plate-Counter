# coding=utf-8
########################
##单双瓦的检测，侧重在滤波
##可以尝试用模板匹配,可能会更稳定
########################
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy

def readimg(imgpath,show=True):
    '''
    读取图片
    转到固定分辨率1440*2560,保持高不变就对了
    '''
    # # 读取输入图片
    # imgpath_gbk = imgpath.encode('gbk')        # unicode转gbk，字符串变为字节数组
    # img = cv2.imread(imgpath_gbk.decode('gbk'))
    # img = cv2.imread(imgpath)
    # img = cv2.imread("data\data\IMG_2155_341.JPG")
    raw_img=cv2.imdecode(np.fromfile(imgpath,dtype=np.uint8),-1)
    # 可截取图片
    # img0=img0[:,:]
    # 将彩色图片转换为灰度图片
    img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)
    h,w=img.shape
    new_h,new_w=2560,1440
    scale=new_h/h
    img = cv2.resize(img,[int(w*scale),new_h])
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) #图像归一化提高对比度，可选参数 cv2.NORM_INF cv2.NORM_L1 cv2.NORM_L2
    if show:
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return raw_img,img,scale


def gaborFilter(img, ksize=20, sigma=1.0, theta=0, lambd=np.pi/2.0, gamma=0.5):
    '''
    garbor 滤波
    波长 (λ)：表示 Gabor 核函数中余弦函数的波长参数。9
    方向 (θ)：表示 Gabor 滤波核中平行条带的方向。[0,360]
    相位偏移 (φ)：表示 Gabor 核函数中余弦函数的相位参数。
    长宽比 (γ)：空间纵横比，决定了 Gabor 函数形状的椭圆率。
    带宽 (b)：Gabor 滤波器的半响应空间频率带宽 b
    '''
    kern = cv2.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
    accum = np.zeros_like(img)
    accum = np.maximum(accum, fimg, accum)
    accum = np.asarray(accum)
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(accum, cmap = 'gray')
    plt.title('Result'), plt.xticks([]), plt.yticks([])
    plt.show()
    return accum

def fourierFliter(img,scale,show=True):
    '''
    傅里叶滤波,保留纵向波形
    scale是读取时的缩放尺度
    oooxxxxooo
    oooxxxxooo
    oooooooooo
    oooxxxxooo
    oooxxxxooo
    '''
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # 得到灰度图能表示的形式,中间是低频，四周是高频
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    if show:
        plt.subplot(121),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()
    rows, cols = img.shape
    crow, ccol = int(rows/2) , int(cols/2)     # 中心位置
    # 低通滤波：输出细节特征，轮廓就会显得模糊，创建掩码时需要中心化为1，周围为0的一个mask矩阵。
    mask = np.zeros((rows, cols, 2), np.uint8)  
    #设置自适应低通滤波宽度 调纵向细节 纵向保留薄板的中频分量，应该保留纵向的全部高频细节，其他纵向的处理应该交到特征统计环节
    col_len1= int(rows/2)  
    col_len2=0 #int(1*rows/64)#0  #int(rows/32) 纵向的低频分量去除，避免后续二值化明暗影响
    row_len1=int(15/scale) #int(rows/8) #调横向细节 横向消除高频分量，保留的低频分量保证检测到边缘，应该根据图片大小自适应
    row_len2=15
    mask[crow-col_len1:crow+col_len1,ccol-row_len1:ccol+row_len1] = 1
    mask[crow-col_len2:crow+col_len2:,ccol-row_len2:ccol+row_len2] = 0
    fshift = dft_shift*mask    # IDFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    if show:
        plt.subplot(121),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
        plt.title('Result'), plt.xticks([]), plt.yticks([])
        plt.show()
        # cv2.imshow('img2',img_back)
    return img_back

def feat_Sats(feat_map,window=[0,100], show=True):
    '''
    统计特征曲线
    feat_map:特征图
    window:采用窗口范围,应该可以把上下的梯度都加一下
    '''
    feat_curve = []
    for i in range(feat_map.shape[0]):      
        feat_curve.append(int(np.floor(sum(feat_map[i, window[0]:window[1]] / (window[1]-window[0]))))) 
    max_ = max(feat_curve)
    for i in range(len(feat_curve)):
        feat_curve[i] = feat_curve[i] / max_ #归一化灰度
    if show:
        plt.plot(range(len(feat_curve)),feat_curve)
        # plt.savefig('my_counter/example.png')
        plt.show()
    return feat_curve


def thr_Counter(feat_curve):
    '''
    基于阈值的峰值计数
    parm fea_curve:特征曲线
    return: 波峰所在索引值的列表
    '''
    result=[]
    peak=[]
    peakWidth =  [9,25]    #len(list_)/20  # 峰的宽度范围，用于排除临近的次峰值，和远离纸堆的噪声
    id = 0  # 标记最近的峰位置
    for i in range(1, len(feat_curve)-1):
        if feat_curve[i-1] < feat_curve[i] and feat_curve[i] > feat_curve[i+1] and feat_curve[i]>0.15:
            peak.append(i)
    for j in range(len(peak)):
        if j==0: 
            if peak[j]-peak[j+1]<peakWidth[1]:
                result.append(peak[j])
        # elif j==len(peak)-1:
        #     if peak[j]-peak[j+1]>peakWidth[0]:
                # result.append(peak[j])
        elif peak[j]-peak[j-1]<peakWidth[1] and peak[j]-peak[j-1]>peakWidth[0]: #过滤掉单峰值和次峰值
            result.append(peak[j])
    return result

def AMPD(data):
    """
    实现AMPD波峰检测算法
    :param data: 1-D numpy.ndarray 
    :return: 波峰所在索引值的列表
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = len(data)
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    return np.where(p_data == max_window_length)[0]

def LSD(img,show=True):
    """
    LSD直线检测
    """
    # 创建一个LSD对象
    lsd = cv2.createLineSegmentDetector(0)
    # 执行检测结果
    dlines = lsd.detect(img)
    # 绘制检测结果
    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(img, (x0, y0), (x1,y1), (0,255,0), 1, cv2.LINE_AA)
    # 显示并保存结果
    # cv2.imwrite('test3_r.jpg', img0)
    if show:
        cv2.imshow("LSD", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 

def savgolFilter(feat_curve,window_length=10, polyorder=2,mode= 'nearest'):
    '''
    曲线拟合滤波,去除毛刺
    '''
    feat_curve=savgol_filter(feat_curve, window_length, polyorder, mode= 'nearest')
    return feat_curve

def gradFeat(img,show=True):
    '''
    提取图像的梯度特征，边缘特征，下面的网址中的算子都试试
    https://blog.csdn.net/m0_38007695/article/details/112796821
    '''
    feat_map = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    #canny = cv2.Canny(img, 50, 150)
    #gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    #dst = cv2.convertScaleAbs(gray_lap) # 转回uint8
    #
    # x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    # y = cv2.Sobel(img, cv2.CV_16S, 0, 1)    
    # absX = cv2.convertScaleAbs(x)# 转回uint8
    # absY = cv2.convertScaleAbs(y)
    # dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    if show:
        cv2.imshow("feat_map", feat_map)
    return feat_map

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    "Marcos Duarte, https://github.com/demotu/BMC"
    [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])
    return ind

def draw_result(raw_img,results,feat_curve,scale,detect=False):
    '''
    把结果画出来
    '''
    colors=["red","green","blue"]
    colors2=[(0,0,255),(0,255,0),(255,0,0)]
    if not detect:
        plt.figure()
        plt.plot(range(len(feat_curve)), feat_curve)
        for i,result in enumerate(results):
            color=colors[i%len(colors)]
            # color2=colors2[i//len(colors)]
            for j,pot in enumerate(result):
                plt.scatter(pot, feat_curve[pot], color=color)
                # if j%5==0:
                #     cv2.line(raw_img, (0,int(pot/scale)), (40,int(pot/scale)),color2,3)               
                #     cv2.putText(raw_img, str(j), (50,int(pot/scale)), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 2)
                # else:
                #     cv2.line(raw_img, (0,int(pot/scale)), (20,int(pot/scale)),color2,2)
                #     # cv2.putText(raw_img, str(j+1), (50,int(pot/scale)), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 2)    
        plt.show()
    else:
        for i,result in enumerate(results):
            color2=colors2[i%len(colors)]
            for j,pot in enumerate(result):
                if j%5==0:
                    cv2.line(raw_img, (0,int(pot/scale)), (40,int(pot/scale)),color2,3)               
                    cv2.putText(raw_img, str(j), (50,int(pot/scale)), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 2)
                else:
                    cv2.line(raw_img, (0,int(pot/scale)), (20,int(pot/scale)),color2,2)
                    # cv2.putText(raw_img, str(j+1), (50,int(pot/scale)), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 2)    
    return raw_img
    

def hist_Sats(filter_img):
    hist = cv2.calcHist([filter_img], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("gray Level")
    plt.ylabel("number of pixels") 
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()
    return 0

def detect(imgpath,thr_wid=None):
    raw_img,img,scale=readimg(imgpath,False)   #读取图片并返回缩放尺度   
    ##傅里叶低通滤波，把纵向的纹理消除,scale调节保留的频率
    filter_img=fourierFliter(img,scale,False)
    filter_img = cv2.normalize(filter_img,None,0,255,cv2.NORM_MINMAX) #图像归一化，可选参数 cv2.NORM_INF cv2.NORM_L1 cv2.NORM_L2
    ##计算垂直梯度
    feat_map=gradFeat(img,False)
    w, h = feat_map.shape
    ##统计特征，单区域简单加和
    feat_curve=feat_Sats(feat_map,[0,int(100*scale)],False)
    #傅里叶尝试得到自适应波长阈值，得到的阈值可用于曲线平滑，峰值检测，结果过滤
    fre_map=np.fft.fft(feat_curve)
    # plt.figure() 
    # plt.plot(fre_map)
    # plt.show()
    thr_wid=int(1900/np.argwhere(fre_map==max(fre_map[9:1280],key=abs)))-1 #经验公式
    ## savgol_filter滤波
    feat_curve = savgolFilter(feat_curve,thr_wid, 2)
    ##峰值检测3
    result_raw=detect_peaks(feat_curve, mph=max(feat_curve)*0.10, mpd=thr_wid, threshold=0, edge='rising')
    ## 结果过滤 ##
    result=[]
    result_temp=[]
    # result=list(filter(lambda x:feat_curve[x]>0.25,result)) # 过滤掉噪点
    result_temp.append(result_raw[0])
    for i in range(1,len(result_raw)):  # 从上到下扫瞄，检测分层
        if result_raw[i]-result_temp[-1]>2*thr_wid+4: #两倍波峰阈值作为分层阈值
            result.append(result_temp)
            result_temp=[result_raw[i]]
        else:
            result_temp.append(result_raw[i])
    result.append(result_temp)
    result=list(filter(lambda x:len(x)>3,result))
    #绘图展示,将检测结果画在图上，并把特征曲线画在旁边
    detected_img=draw_result(raw_img,result,feat_curve,scale,True)
    # cv2.imshow("detected_img", detected_img)
    #把检测到的结果画到图上
    return result,detected_img

if __name__=="__main__":
    #后面的数值表示波峰宽度和频域极大值对应频率之间的关系，从而进行自适应
    # imgpath="data\data\\5.png" # 22 220 双层
    # imgpath="data\data\\2.jpg" # 417 4-6
    # imgpath="data\data\IMG_2155_341.JPG"  #351  4-7
    # imgpath="data\data\IMG_2153_26.JPG"  # 26 90-100 
    imgpath="data\data\IMG_2156_381.JPG" # 380 5-8
    # imgpath="data\data\IMG_2160_400.JPG"#402 3-5
    # imgpath="data\data\\20230530143449_26.jpg" #106 20-25
    # imgpath="data\data\IMG_2207_263.JPG" #287 9
    # imgpath="data\data\cuted_IMG_2209_261or264.JPG"#厚薄双层292,540 9,4 不要设阈值通过间隔过滤就可以了，多个断层就输出不同断层结果
    # imgpath="data\data\IMG_2207_new263.JPG" #268 7-9
    # imgpath="data\data\IMG_2008_1.JPG" #188 11-12
    # imgpath="data\data\\aaac8830d147d2538a1ac2fdadee36a3.JPG"
    # imgpath="data\data\IMG_2008_2.JPG"#
    # imgpath="data\data\ea9e927e6b332870c8e652d43a5ec90e.JPG"
    # imgpath="data\data\IMG_2216.JPG"
    # imgpath="data\data\IMG_2211.JPG" # 这张图的问题在于切面的梯度特征比侧面高出太多了
    raw_img,img,scale=readimg(imgpath,False)   #读取图片并返回缩放尺度   
    #################################### 滤波 #################################
    #图片锐化
    # sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    # sharpen_image = cv2.filter2D(img, cv2.CV_32F, sharpen_op)
    # sharpen_image = cv2.convertScaleAbs(sharpen_image)
    # cv2.imshow("sharpen_image", sharpen_image)
    ##高斯模糊，水平均值滤波
    # sharpen_op = np.array([[0,0, 0], [1/3, 1/3, 1/3], [0, 0, 0]], dtype=np.float32) #水平
    # sharpen_image = cv2.filter2D(img, cv2.CV_32F, sharpen_op)
    # img = cv2.convertScaleAbs(sharpen_image)
    # cv2.imshow("sharpen_image", img)
    ##边缘滤波
    # sharpen_op = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) #水平
    # sharpen_image = cv2.filter2D(img, cv2.CV_32F, sharpen_op)
    # img = cv2.convertScaleAbs(sharpen_image)
    # cv2.imshow("sharpen_image", img)
    ##傅里叶低通滤波，把纵向的纹理消除,scale调节保留的频率
    filter_img=fourierFliter(img,scale,True)
    filter_img = cv2.normalize(filter_img,None,0,255,cv2.NORM_MINMAX) #图像归一化，可选参数 cv2.NORM_INF cv2.NORM_L1 cv2.NORM_L2
    ##gabor滤波
    # filter_img=gaborFilter(img, ksize=20, sigma=1.0, theta=0, lambd=np.pi*2.0, gamma=0.5)
    ################################# 特征增强 ######################################
    ## 图片锐化
    ## meanshift
    ## 对比度增强
    # cv2.imshow("img", img)
    ##灰度值统计
    # hist_Sats(filter_img)
    # dst = np.zeros_like(src)
    # cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
    #################################### 直线检测 ################################
    ##lsd
    # iff_img=iff_img.astype(np.uint8)
    # LSD(iff_img,show=True)
    ################################### 特征提取 ####################################
    ##计算垂直梯度
    feat_map=gradFeat(img,False)
    # feat_map=np.exp(feat_map)
    ##自相关特征
    ##傅里叶分析
    ##局部二值化 效果不好
    # feat_map = cv2.normalize(feat_map,None,0,255,cv2.NORM_MINMAX)
    # feat_map=np.floor(feat_map)
    # feat_map=feat_map.astype(np.uint8)
    # thr, feat_map = cv2.threshold(feat_map, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #全局
    # feat_map = cv2.adaptiveThreshold(feat_map, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)  #局部
    # cv2.imshow('hh',feat_map)    
    ################################### 统计特征曲线 #################################
    w, h = feat_map.shape
    ##单区域简单加和
    feat_curve=feat_Sats(feat_map,[0,int(100*scale)],True)
    ##多区域采样，左中右三个区域,短而多可以保证准确率,倾斜的也可以,后续检测到峰值后，做峰值对齐，避免漏检
    # feat_curve2=feat_Sats(feat_map,[int(w/2)-50,int(w/2)+50])
    # feat_curve3=feat_Sats(feat_map,[w-100,w])
    ##################################### 特征曲线处理 ###################################
    #傅里叶尝试得到自适应波长阈值，得到的阈值可用于曲线平滑，峰值检测，结果过滤
    fre_map=np.fft.fft(feat_curve)
    plt.figure() 
    plt.plot(fre_map)
    plt.show()
    thr_wid=int(h/np.argwhere(fre_map==max(fre_map[9:1280],key=abs))) #经验公式
    ##用滑动窗口做非极大值抑制,把窗口内的非极大值都置0

    ## savgol_filter滤波
    feat_curve = savgolFilter(feat_curve,int(thr_wid/2), 2)
    ##均值滤波
    # feat_curve=np.convolve(feat_curve, np.ones((3,))/3, mode="same")
    # 推土滤波，想象一把瓦刀，抹平所有缝隙
    ##数据截断
    # feat_curve = np.clip(feat_curve, 0, None)
    ##################################### 周期波峰检测计数 ##################################
    ##基于阈值的计数，
    # result=thr_Counter(feat_curve)
    ##AMPD波峰检测算法
    # result=AMPD(feat_curve)
    ##峰值检测1
    # result = scipy.signal.find_peaks_cwt(feat_curve, 3)
    ##峰值检测2
    # result = scipy.signal.argrelextrema(feat_curve, np.greater, order=8)[0]
    ##峰值检测3
    result_raw=detect_peaks(feat_curve, mph=max(feat_curve)*0.10, mpd=thr_wid, threshold=0, edge='rising')
    ######################################### 结果过滤 ############################################
    result=[]
    result_temp=[]
    # result=list(filter(lambda x:feat_curve[x]>0.25,result)) # 过滤掉噪点
    result_temp.append(result_raw[0])
    for i in range(1,len(result_raw)):  # 从上到下扫瞄，检测分层
        if result_raw[i]-result_temp[-1]>2*thr_wid+4: #两倍波峰阈值作为分层阈值
            result.append(result_temp)
            result_temp=[result_raw[i]]
        else:
            result_temp.append(result_raw[i])
    result.append(result_temp)
    result=list(filter(lambda x:len(x)>3,result))
    ########################################### 结果展示 ###########################################
    #绘图展示,将检测结果画在图上，并把特征曲线画在旁边
    detected_img=draw_result(img,result,feat_curve,scale,False)
    cv2.imshow("detected_img", detected_img)
    #把检测到的结果画到图上
    for ans in result:
        print(len(ans)-1)  #输出结果
    #基于傅立叶变换提取曲线特征的自适应波峰检测方法
    #idea：基于CLAHE(Contrast Limited Adaptive Histogram Equalization)原理、二维均值滤波及形态学膨胀腐蚀优化原理,