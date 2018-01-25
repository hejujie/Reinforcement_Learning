import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate = 0.01,
                 reward_decay = 0.9,
                 replace_target_iter = 300,
                 memory_size = 500,
                 e_greedy = 0.9,
                 batch_size = 32,
                 e_greedy_increment = None,
                 output_graph = False,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        
        #记录学习次数（用于判断是否更新）
        self.learn_step_counter = 0
        
        #初始化记忆体，使用numpy，两个状态，一个reward，一个done。
        self.memory = np.zeros((self.memory_size, n_features*2+2))

        #创建网络：
        self._build_net()
        self.sess = tf.Session()
        
        #输出tensorbroad：
        if output_graph:
        # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
    
    def _build_net(self):      
    #eval network, 这个是更新的网络。
        #用于接受游戏中的state
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name = 's')
        #接受来自另一个网络的Q_target，用于计算loss。
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name = 'Q_target')
        
        #使用C_name作为collection，存储参数。
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)
            
            #eval网络的第一层。
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer = w_initializer, collections = c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            
            #eval网络的第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
                
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            
    #target network：这个是用于比较的网络，固定次数进行更新
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name = 's_')
        with tf.variable_scope('target_net'):
            c_names, n_l1, w_initializer, b_initializer = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)
            
            #target网络第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer = w_initializer, collections = c_names)
                b1 = tf.get_variable('b1', shape = [1, n_l1], initializer=w_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
                
            #target网络第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w1', shape = [n_l1, self.n_actions], initializer = w_initializer, collections=c_names)
                b1 = tf.get_variable('b2', shape = [1, self.n_actions], initializer=w_initializer, collections= c_names)
                self.q_next = tf.matmul(l1, w2) + b2
    
    
    #用于存储状态
    def store_transition(self, s, a, r, s_):
        #hasattr表示确定一个对象是否有某个属性，这里如果没有memory_counter 那么就添加它。
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
            
        #一个状态：
        transition = np.hstack((s, [a, r], s_))
        
        #判断是否越界，如果越界，那么就从头开始，相当于循环。
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        
        self.memory_counter += 1
        
    #选择一个行为：
    def choose_action(self, observation):
        #将格式变成成[1, size_of_observaion]， newaxis相当于新加了一个周
        observation = observation[np.newaxis, :]
        
        #self.s: 表示传入的是一个怎么样的值
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict = {self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
            
        return action
        
    #使用eval的参数替换target的参数（固定step后）
    def _replace_target_params(self):
        #变成了一个列表，进行列表操作，使用tf.assign赋值
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        
    #进行学习
    def learn(self):
        #替换参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\n target_params_replace\n')
            
        #size表示随机数的结果的结构， 前面表示在某个范围里面取。
        #如果少了，那么就会重复，但是关系不大。
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size = self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size = self.batch_size)
        #按照下表取对应行的值
        batch_memory = self.memory[sample_index, :]
        
        #分别得到两个网络中对应的replay的Q值。
        #注意这里的Q是包含当前状态的所有action的Q的，所以在后续需要进行处理，之前选择的action的Q值。
        #细节:这里的-：和：表示最前面何最后面
        q_next, q_eval = self.sess.run(
        [self.q_next, self.q_eval],
        feed_dict = {
            self.s_: batch_memory[:, -self.n_features:],
            self.s : batch_memory[:, :self.n_features]
            }
        )
        
        #这里进行的处理是首先复制了eval给target，然后根据action的选择，将q_next对应项的Q进行加法处理，之后与——next
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype = np.int32)
        #batch中所有的行对应的action的下表。
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        #对target中的对应的每一行的对应列进行赋值，即memory-action的值变成q_next中的max。
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis = 1)
        
        
        #这时，只要再次减去q_eval就可以把剩余的行清零了。
        #传入Q-target，剩余的q_eval自带计算，计算得到cost，并进行优化调参。 
        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict = {self.s: batch_memory[:, :self.n_features], self.q_target: q_target})
        
        self.cost_his.append(self.cost)
        
        #逐步增加epislon，降低随机性。
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        
        
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('cost')
        plt.xlabel('training steps')
        plt.show()
        
        
                