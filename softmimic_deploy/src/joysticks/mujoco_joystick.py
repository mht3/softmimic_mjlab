import time
import numpy as np

class MujocoJoystick:
    
    def __init__(self):
        self.mode = 0
        self.ctrlmode_left = 0
        self.ctrlmode_right = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0
        self.current_policy = 1
        self.running = False
        self.run_thread = None
        self.root = None
        self.step_counter = 0
        
        self.a_button, self.b_button, self.x_button, self.y_button = 0, 0, 0, 0

        # smoothing
        self.left_stick_smooth = [0, 0]
        self.right_stick_smooth = [0, 0]
    
    def key_callback(self, key):
        """
        Key callback function.
        - window: the window that received the event
        - key: the keyboard key that was pressed or released
        - scancode: the system-specific scancode of the key
        - action: GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT
        - mods: bitfield describing which modifier keys were held down
        """
        print(key)
        if key == 32: # SPACE
            # self.right_lower_right_switch_pressed = 1 - self.right_lower_right_switch_pressed
            # self.right_lower_right_switch = 1 - self.right_lower_right_switch
            self.left_stick[0] = 0
            self.left_stick[1] = 0
            self.right_stick[0] = 0
            self.right_stick[1] = 0
        elif key == 265: # UP
            self.left_stick[1] += 0.2
        elif key == 264: # DOWN
            self.left_stick[1] -= 0.2
        elif key == 263: # LEFT
            self.right_stick[0] -= 0.2
        elif key == 262: # RIGHT
            self.right_stick[0] += 0.2
        elif key == 44: # ,
            self.current_policy = 1
        elif key == 46: # .
            self.current_policy = 2
        elif key == 266: # page up
            self.right_stick[1] += 0.1
        elif key == 267: # page down
            self.right_stick[1] -= 0.1
        # A button
        elif key == 65:
            self.a_button = 1
        # B button
        elif key == 66:
            self.b_button = 1
        # X button
        elif key == 88:
            self.x_button = 1
        # Y button
        elif key == 89:
            self.y_button = 1
        # L1 button
        elif key == 76: # L
            # self.left_upper_switch_pressed = 1 - self.left_upper_switch_pressed
            self.left_upper_switch = 1 #- self.left_upper_switch
    
    def update_stick(self, stick, x, y):
        if stick == 'left':
            self.left_stick = [x, y]
        elif stick == 'right':
            self.right_stick = [x, y]
    
    def update_switch(self, switch, state):
        if switch == 'left_upper':
            self.left_upper_switch = state
        elif switch == 'right_upper':
            self.right_upper_switch = state
        elif switch == 'left_lower':
            self.left_lower_left_switch = state
            self.left_lower_left_switch_pressed = state
        elif switch == 'right_lower':
            self.right_lower_right_switch = state
            self.right_lower_right_switch_pressed = state
    
    def get_buttons(self):
        buttons = np.array([self.left_lower_left_switch, self.left_upper_switch, self.right_lower_right_switch, self.right_upper_switch])
        self.left_lower_left_switch, self.left_upper_switch, self.right_lower_right_switch, self.right_upper_switch = 0, 0, 0, 0
        return buttons


    def get_abxy(self):
        abxy = np.array([self.a_button, self.b_button, self.x_button, self.y_button])
        self.a_button, self.b_button, self.x_button, self.y_button = 0, 0, 0, 0
        return abxy
    
    def get_command(self):
        # self.left_stick_smooth = [0.8 * self.left_stick_smooth[0] + 0.2 * self.left_stick[0], 0.8 * self.left_stick_smooth[1] + 0.2 * self.left_stick[1]]
        # self.right_stick_smooth = [0.8 * self.right_stick_smooth[0] + 0.2 * self.right_stick[0], 0.8 * self.right_stick_smooth[1] + 0.2 * self.right_stick[1]]

        # return [self.left_stick_smooth[1], self.left_stick_smooth[0], self.right_stick_smooth[0], self.right_stick_smooth[1]]

        return [self.left_stick[1], self.left_stick[0], self.right_stick[0], self.right_stick[1]]

    def get_current_policy(self):
        return self.current_policy