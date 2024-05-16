#!/c/bit9prog/dev/anaconda3/python
#!/usr/bin/env python

import pywin
# from pywin32 import win32api
import win32api, win32gui, win32con
import time, datetime
import random
import sys

def windowEnumerationHandler(hwnd, top_windows): #https://github.com/iwalton3/plex-mpv-shim/blob/master/plex_mpv_shim/win_utils.py
    top_windows.append((hwnd, win32gui.GetWindowText(hwnd)))

def findShowWin(keyWord):
    top_windows = []
    fg_win = win32gui.GetForegroundWindow()
    win32gui.EnumWindows(windowEnumerationHandler, top_windows)
    # print(len(top_windows))
    for i in top_windows:
        if len(i[1]):
            # print(f"WinTitle={i[1]}")
            if keyWord in i[1]: #.lower():
                # if i[0] != fg_win:
                win32gui.ShowWindow(i[0], win32con.SW_MINIMIZE) # Minimize,   6
                # win32gui.ShowWindow(i[0], win32con.SW_RESTORE)  # Un-minimize,9
                win32gui.ShowWindow(i[0], win32con.SW_MAXIMIZE) # Maximize,
                win32gui.SetActiveWindow(i[0])
                break

def flashCursor():
    # win32gui.ShowCaret(None)
    for i in range(10):
        dx=random.randint(-10, 10)
        dy=random.randint(-10, 10)
        win32api.ShowCursor(False)
        time.sleep(random.randint(2, 10)*.25)
        win32api.ShowCursor(True)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE_NOCOALESCE | win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

def MoveMouse(x, y):
    # if self.run_time.stop: return
    x, y = int(x), int(y)
    # self.root.debug("Tapping at location ({},{})".format(x, y))
    # if self._debug: input("waiting for confirmation press enter") # Helper to debug taps
    # ox, oy = win32api.GetCursorPos()
    try:
        curr_window = win32gui.GetForegroundWindow()
        # win32gui.ShowWindow(curr_window, win32con.SW_MINIMIZE)
        # x, y = int(x), int(y)
        cx, cy = win32gui.ClientToScreen(curr_window, (x, y))
        #x, y = self.__calculate_absolute_coordinates__(cx, cy)
        # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, x, y, 0, 0)
        flashCursor()
    except Exception as e:
        print("Exception: ", e)


    # win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, cx, cy, 0, 0)
    # win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, cx, cy, 0, 0)
    time.sleep(random.randint(2, 10)) #2-10 seconds
    # win32api.SetCursorPos((ox, oy))
    # win32gui.ShowWindow(curr_window, win32con.SW_RESTORE)
    # win32gui.SetActiveWindow(curr_window)

def randMoveMouse():
    x,y=win32api.GetCursorPos()
    # dx=random.randint(-25, 25)
    # dy=random.randint(-25, 25)
    MoveMouse(x,y)

def main():
    winTitleKeyWord='Umap - Zotero'
    if len(sys.argv)>1: winTitleKeyWord= sys.argv[1]
    nMove=80 #on average, about 8 -> 1 hour since 60/7.5=8
    if len(sys.argv)>2: nMove=int(sys.argv[2])

    for t in range(nMove): #100*2min=3 hours
        nmin=random.randint(5, 10)
        print(f"{datetime.datetime.now()}: moveMouse and sleep {nmin} minutes")
        origWin=win32gui.GetForegroundWindow()
        # win32gui.ShowWindow(origWin, win32con.SW_MINIMIZE)

        findShowWin(winTitleKeyWord)
        randMoveMouse()

        #origWin might vanish
        try:
            #win32gui.ShowWindow(origWin, win32con.SW_MAXIMIZE)
            win32gui.SetActiveWindow(origWin)
        except Exception as e:
            print("Exception: ", e)


        time.sleep(60 * nmin+random.randint(0,50))

if __name__ == "__main__": main()
