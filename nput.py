import pynput
from pynput.keyboard import Key,Controller
keyboard=Controller()
while(1):
    keyboard.press(Key.space)
    keyboard.release(Key.space)
