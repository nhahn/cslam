#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

from sortedcontainers import SortedDict

from cslam.vpr.cosplace import CosPlace

from Yolov7_StrongSORT_OSNet.msg import CroppedObject, CroppedObjectArray, KeyframeOdomRGB
from cslam_common_interfaces.msg import MOTGlobalDescriptor, MOTGlobalDescriptors

class CroppedDescriptorGraph(object): 
    def __init__(self, params, node):
        """Initialization

        Args:
            params (dict): parameters
            node (ROS 2 node handle): node handle
        """
        self.params = params
        self.node = node
                
        # Fix node name
        # Gets this every 7 seconds from strongsort module
        self.keyframe_info_sub = self.node.create_subscription(
            MOTGlobalDescriptors, 
            'cslam/mot/descriptors',
            self.keyframe_callback, 
            100 
        )
        
        # # TODO Change such that only the robot that published this info gets it
        # self.keyframe_info_sub = self.node.create_subscription(
        #     CroppedObjectArray, 
        #     'cslam/mot/keyframe_rgb_odom',
        #     self.keyframe_callback, 
        #     100 
        # )
        
        # self.mot_global_desc_arr = MOTGlobalDescriptors()
        
        # # Info: (obj_id, (robot_id, keyframe_id, confidence, obj_class_id, odom, embedding))
        # # Does this need to be a SortedDict?
        # self.best_desc_per_object = SortedDict()
        # self.highest_obj_idx = 0
        
        # # Inter-robot communication to construct graph, etc
        # self.mot_global_desc_arr_pub = self.node.create_publisher(
        #     MOTGlobalDescriptors, 
        #     "cslam/mot/global_descriptors", 
        #     100
        # )
        
        # # Inter-robot communication to construct graph, etc
        # self.mot_global_desc_arr_sub = self.node.create_subscription(
        #     MOTGlobalDescriptors, 
        #     "cslam/mot/global_descriptors", 
        #     self.other_global_desc_callback,
        #     100
        # )
        
        # self.construct_graph_list = self.node.create_timer(
        #     7, # chosen arbitrarily to offset the 5 second callback of the SLAM calculations
        #     self.construct_graphs_callback, 
        #     clock=Clock()
        # )
        
        global cv2
        import cv2
        global CvBridge
        from cv_bridge import CvBridge

    # # Use global image similarity as discriminator first, and then odometry information
    # def keyframe_callback(self, msg):         
    #     bridge = CvBridge()
    #     cv_img = bridge.imgmsg_to_cv2(msg.image, desired_encoding='passthrough')
        
    #     for det in msg: # det of type CroppedObject
            
    #         if det.obj_id > self.highest_obj_idx: 
    #             self.form_mot_global_desc(msg, det, cv_img)
    #             self.highest_obj_idx = det.obj_id
    #         else: 
    #             (conf, obj_class_id, _) = self.best_desc_per_object[det.obj_id]
                
    #             if det.confidence > conf and det.obj_class_id == obj_class_id: 
    #                 self.form_mot_global_desc(msg, det, cv_img)
    #             else: 
    #                 continue
            
    #         # mot_embeddings.append((msg.robot_id, det.obj_class_id, det.confidence, obj_embedding))

    #     # self.add_mot_descriptors_to_graphs(mot_embeddings, msg.keyframe_id)
        
    # def form_mot_global_desc(self, msg, det, cv_img): 
    #     cropped = cv_img[det.top_left_x:det.top_left_x + det.width, det.top_left_y:det.top_left_y + det.height]
    #     embedding = self.compact_descriptor.compute_embedding(cropped)
    #     self.best_desc_per_object[det.obj_id] = (
    #         msg.robot_id, 
    #         msg.keyframe_id, 
    #         det.confidence, 
    #         det.obj_class_id, 
    #         det.odom, 
    #         embedding
    #     )
        
    # # every 7 seconds, publish MOTGlobalDescriptors message 
    # # every other robot gets this list
    # # the broker constructs the graph locally, runs nearest neighbor matching and 
    # # odometry estimates, and contains the object tracking version of the 'map reference frame'
    # def share_curr_agent_global_desc(self): 
    #     for (obj_id, (robot_id, keyframe_id, conf, obj_class_id, odom, embedding)) in self.best_desc_per_object.items(): 
    #         desc = MOTGlobalDescriptor(
    #             robot_id = robot_id, 
    #             keyframe_id = keyframe_id, 
    #             obj_id = obj_id, 
    #             obj_class_id = obj_class_id, 
    #             confidence = conf, 
    #             odom = odom, 
    #             descriptor = embedding
    #         )

    #         self.mot_global_desc_arr.descriptors.append(desc)
            
    #     self.mot_global_desc_arr_pub.publish(self.mot_global_desc_arr)
    
    

    # get list of the best descriptors of each detected object in the last 7 seconds from each robot 
    # each robot stores this list
    # only broker needs to construct a graph 
    # 
    # Option A: 
    # - Pairwise similarity between each set of descriptors (optimize somehow if possible) 
    # - Clustering
    # 
    # Option B: 
    # - Edge for each nearest neighbor descriptor 
    # - Broker calculates maximum algebraic connectivity + vertex cover in graph
    #  
    # need to allocate broker... how? 
    # how to actually change the labels? A mapping?
    # This doesn't seem like it'll scale very well, what are some options that don't require  
    def keyframe_callback(self, msg): 
        if msg.descriptors[0].robot_id != self.params['robot_id']: 
            print("TODO complete")
        
        
        
        # matches = self.lcm.add_local_global_descriptor(embedding, keyframe_id)
