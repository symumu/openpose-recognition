import cv2
import time
import numpy as np
from random import randint

# 根据路径输入图片
image1 = cv2.imread("test/1.jpg")

protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"
nPoints = 18

# COCO Output Format    COCO输出格式
# POSE_PAIRS分别代表keypointsMapping里面同一根骨骼两端的两个人体关节(关键点)。
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip',
                    'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
POSE_PAIRS = [[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],
              [1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
              [1,0],[0,14],[14,16],[0,15],[15,17],[2,17],[5,16]]

# index of PAFs correspoding to the POSE_PAIRS
# PAFs与姿态对的相关指数
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
# 例如，对于位姿对(1，2)，PAFs位于输出的指数(31，32)，类似地，(1，5)->(39，40)等等。
# mapIdx：代表与POSE_PAIRS对应的亲和场特征图索引
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]

colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
          [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
          [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

# 找到人体关节点，关节点特征图
def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    # find the blobs  找出关节点
    contours, hierarchy = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
    return keypoints
# Find valid connections between the different joints of a all persons present
# 在所有在场的人的不同关节之间找到有效的连接（PAF）——区分关键点
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # 两个可能连接的关节
    for k in range(len(mapIdx)):
        # A->B构成肢体（亲和场特征图）
        pafA = output[0, mapIdx[k][0], :, :]  # 第k组连接关节的第一个关节PAF
        pafB = output[0, mapIdx[k][1], :, :]  # 第k组连接关节的第二个关节PAF
        pafA = cv2.resize(pafA,(frameWidth, frameHeight))
        pafB = cv2.resize(pafB,(frameWidth, frameHeight))
        # 找到这两个关节的位置（对应肢体关键点索引）
        candA = detected_keypoints[POSE_PAIRS[k][0]]  # 找到第一个关节的位置(所有人)
        candB = detected_keypoints[POSE_PAIRS[k][1]]  # 找到第二个关节的位置(所有人)
        nA = len(candA)
        nB = len(candB)
        # 如果检测到连接对的关键点，用candA中的每一个关节检查candB中的每一个关节，计算两个关节之间的距离向量。
        # 在节点之间的一组插值点处找到PAF值使用上面的公式计算一个分数来标记连接有效。
        # 使用公式计算亲和场的得分
        if(nA != 0 and nB != 0):  # 如果有这两个关节
            valid_pair = np.zeros((0,3))
            for i in range(nA):  # 对于第一个关节的所有人遍历
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):  # 第二个关节的所有人遍历
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)
                    # Check if the connection is valid 检查连接是否有效
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    # 如果与PAF对齐的插值向量的分数高于阈值->有效对
                    if(len(np.where(paf_scores>paf_score_th)[0])/n_interp_samples)>conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list  将连接附加到列表中
                if found:
                    # 被连接的肢体的关键点索引
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)
            # Append the detected connections to the global list 将检测到的连接附加到全局列表
            valid_pairs.append(valid_pair)
        else:   # 如果关节被遮挡等原因，导致不存在
                # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs
# This function creates a list of keypoints belonging to each person  此函数创建属于每个人的关键点列表
# For each detected valid pair, it assigns the joint(s) to a person  对于每个检测到的有效对，它将关节分配给一个人
# 检测到的人体关键点分配到对应的人体上（根据获得的能被连接的关键点对，把坐标也对应好）
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score  每行的最后一个数字是总分
    personwiseKeypoints = -1 * np.ones((0, 19))
    for k in range(len(mapIdx)):    # 遍历有效的关节连接
        if k not in invalid_pairs:  # 当前关节存在
            partAs = valid_pairs[k][:,0]    # 所有人第一个关节索引
            partBs = valid_pairs[k][:,1]    # 遍历有效的关节连接
            indexA, indexB = np.array(POSE_PAIRS[k])    # 遍历有效的关节连接
            for i in range(len(valid_pairs[k])):    # 当前关节有多少个数据点(人)
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):   # 遍历人
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break
                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int),2]+valid_pairs[k][i][2]
                # if find no partA in the subset, create a new subset  如果在子集中找不到partA，则创建一个新子集
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score 将两个关键点的关键点_分数和paf_分数相加
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int),2])+valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints,row])
    return personwiseKeypoints

'''
提取所有的关节的位置和置信度，就相当于把每个关节的特征图遍历一遍：
nPoints = 18
def get_joint_kps(output):
    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoints_id = 0
    threshold = 0.1
    for part in range(nPoints):
        probMap=output[0,part,:,:]
        probMap = cv2.resize(probMap,(img.shape[1],img.shape[0]))
        keypoints = getKeypoints(probMap,threshold)
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i]+(keypoints_id,)) #所有人的18个关节位置、置信度、id
            keypoints_list = np.vstack([keypoints_list,keypoints[i]])
            keypoints_id += 1
        detected_keypoints.append(keypoints_with_id)
    return detected_keypoints,keypoints_list
函数调用    
detected_keypoints,keypoints_list = get_joint_kps(output)
'''

# 图像处理
def bilinear_interpolation(img, out):  # out为希望输出大小
    src_h, src_w, channel = img.shape  # 获取三通道
    dst_h, dst_w = out[1], out[0]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

# 图像输入
image2 = bilinear_interpolation(image1, (550,600))
frameWidth = image2.shape[1]
frameHeight = image2.shape[0]
t = time.time()
# 直接用opencv的dnn.readNetFromCaffe来调用模型
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
# Fix the input Height and get the width according to the Aspect Ratio
# 固定输入高度，并根据纵横比获得宽度
inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)
inpBlob = cv2.dnn.blobFromImage(image2,1.0/255,(inWidth,inHeight),(0,0,0),swapRB=False,crop=False)
net.setInput(inpBlob)
output = net.forward()

print("Time Taken in forward pass = {}".format(time.time() - t))
detected_keypoints = []
keypoints_list = np.zeros((0,3))
keypoint_id = 0
threshold = 0.1

# 提取图像中有几个人
for part in range(nPoints):
    probMap = output[0,part,:,:]
    probMap = cv2.resize(probMap, (image2.shape[1], image2.shape[0]))
    keypoints = getKeypoints(probMap, threshold)
    # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
    if keypointsMapping[part]=='Nose':
        print(len(keypoints))
    keypoints_with_id = []
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
        keypoints_list = np.vstack([keypoints_list, keypoints[i]])
        keypoint_id += 1
    detected_keypoints.append(keypoints_with_id)
# print(len(keypoints))

# 输出函数
frameClone = image2.copy()

# 关键点可视化
for i in range(nPoints):
    for j in range(len(detected_keypoints[i])):
        cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
# cv2.imshow("Keypoints",frameClone)
valid_pairs, invalid_pairs = getValidPairs(output)
personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

# 人肢体可视化
for i in range(17):
    for n in range(len(personwiseKeypoints)):
        index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
        if -1 in index:
            continue
        B = np.int32(keypoints_list[index.astype(int), 0])
        A = np.int32(keypoints_list[index.astype(int), 1])
        cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
'''
cv2.imshow("Openpose using OpenCV",frameClone)
cv2.waitKey(0)
'''