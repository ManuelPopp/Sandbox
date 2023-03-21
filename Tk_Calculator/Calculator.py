#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:27:06 2023
"""
__author__ = "Manuel"
__date__ = "Mon Mar 20 18:27:06 2023"
__credits__ = ["Manuel R. Popp"]
__license__ = "Unlicense"
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Production"

import re, pyperclip
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import N, S, E, W
from numpy import sqrt, log, log10, linspace
from random import choice

#-----------------------------------------------------------------------------|
# Functions
def set(variable, value):
    variable = value

#-----------------------------------------------------------------------------|
# Window design
class Calculator(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.pack(padx = 1, pady = 1)
        self.createWidgets()
        self.master.minsize(200, 175)
        
        # Plot options
        self.x_min = tk.DoubleVar()
        self.x_max = tk.DoubleVar()
        self.n_ticks = tk.IntVar()
        
        self.x_min.set(-10.)
        self.x_max.set(10.)
        self.n_ticks.set(100)
    
    def createWidgets(self):
        # Variables
        self.value = tk.StringVar()
        self.value.set("")
        self.funct = tk.BooleanVar()
        self.funct.set(False)
        self.error = tk.BooleanVar()
        self.error.set(False)
        self.b_last_pressed = tk.StringVar()
        self.b_last_pressed.set("")
        
        self.winfo_toplevel().title("PyCalc")
        
        # Display
        self.display = tk.Entry(self)
        self.display.grid(row = 0, column = 0, columnspan = 5, sticky = "EW")
        self.display["textvariable"] = self.value
        
        # Buttons
        ## Function plotting
        self.fun = tk.Button(self)
        self.fun["text"] = "f(x)="
        self.fun["command"] = self.function_start
        self.fun.grid(row = 1, column = 0, sticky = "EW")
        
        self.bx = tk.Button(self)
        self.bx["text"] = "X"
        self.bx["command"] = lambda: self.b_pressed("x")
        self.bx.grid(row = 1, column = 1, sticky = "EW")
        
        self.edit_plot_opts = tk.Button(self)
        self.edit_plot_opts["text"] = "Options"
        self.edit_plot_opts["command"] = self.edit_plot_options
        self.edit_plot_opts.grid(row = 1, column = 2, columnspan = 2,
                                 sticky = "EW")
        
        ## Delete button
        self.bdel = tk.Button(self)
        self.bdel["text"] = "DEL"
        self.bdel["command"] = self.delete
        self.bdel.grid(row = 1, column = 4, sticky = "EW")
        
        ## Calculations
        self.b7 = tk.Button(self)
        self.b7["text"] = "7"
        self.b7["command"] = lambda: self.b_pressed("7")
        self.b7.grid(row = 2, column = 0, sticky = "EW")
        
        self.b8 = tk.Button(self)
        self.b8["text"] = "8"
        self.b8["command"] = lambda: self.b_pressed("8")
        self.b8.grid(row = 2, column = 1, sticky = "EW")
        
        self.b9 = tk.Button(self)
        self.b9["text"] = "9"
        self.b9["command"] = lambda: self.b_pressed("9")
        self.b9.grid(row = 2, column = 2, sticky = "EW")
        
        self.bsq = tk.Button(self)
        self.bsq["text"] = "^"
        self.bsq["command"] = lambda: self.b_pressed("**")
        self.bsq.grid(row = 2, column = 3, sticky = "EW")
        
        self.bclr = tk.Button(self)
        self.bclr["text"] = "CLEAR"
        self.bclr["command"] = self.clear
        self.bclr.grid(row = 2, column = 4, sticky = "EW")
        
        self.b4 = tk.Button(self)
        self.b4["text"] = "4"
        self.b4["command"] = lambda: self.b_pressed("4")
        self.b4.grid(row = 3, column = 0, sticky = "EW")
        
        self.b5 = tk.Button(self)
        self.b5["text"] = "5"
        self.b5["command"] = lambda: self.b_pressed("5")
        self.b5.grid(row = 3, column = 1, sticky = "EW")
        
        self.b6 = tk.Button(self)
        self.b6["text"] = "6"
        self.b6["command"] = lambda: self.b_pressed("6")
        self.b6.grid(row = 3, column = 2, sticky = "EW")
        
        self.mult = tk.Button(self)
        self.mult["text"] = "×"
        self.mult["command"] = lambda: self.b_pressed("*")
        self.mult.grid(row = 3, column = 3, sticky = "EW")
        
        self.div = tk.Button(self)
        self.div["text"] = "/"
        self.div["command"] = lambda: self.b_pressed("/")
        self.div.grid(row = 3, column = 4, sticky = "EW")
        
        self.b1 = tk.Button(self)
        self.b1["text"] = "1"
        self.b1["command"] = lambda: self.b_pressed("1")
        self.b1.grid(row = 4, column = 0, sticky = "EW")
        
        self.b2 = tk.Button(self)
        self.b2["text"] = "2"
        self.b2["command"] = lambda: self.b_pressed("2")
        self.b2.grid(row = 4, column = 1, sticky = "EW")
        
        self.b3 = tk.Button(self)
        self.b3["text"] = "3"
        self.b3["command"] = lambda: self.b_pressed("3")
        self.b3.grid(row = 4, column = 2, sticky = "EW")
        
        self.add = tk.Button(self)
        self.add["text"] = "+"
        self.add["command"] = lambda: self.b_pressed("+")
        self.add.grid(row = 4, column = 3, sticky = "EW")
        
        self.sub = tk.Button(self)
        self.sub["text"] = "-"
        self.sub["command"] = lambda: self.b_pressed("-")
        self.sub.grid(row = 4, column = 4, sticky = "EW")
        
        self.b0 = tk.Button(self)
        self.b0["text"] = "0"
        self.b0["command"] = lambda: self.b_pressed("0")
        self.b0.grid(row = 5, column = 0, sticky = "EW")
        
        self.dot = tk.Button(self)
        self.dot["text"] = "."
        self.dot["command"] = lambda: self.b_pressed(".")
        self.dot.grid(row = 5, column = 1, sticky = "EW")
        
        self.exp = tk.Button(self)
        self.exp["text"] = "^"
        self.exp["command"] = lambda: self.b_pressed("**")
        self.exp.grid(row = 5, column = 2, sticky = "EW")
        
        self.brack1 = tk.Button(self)
        self.brack1["text"] = "("
        self.brack1["command"] = lambda: self.b_pressed("(")
        self.brack1.grid(row = 5, column = 3, sticky = "EW")
        
        self.brack0 = tk.Button(self)
        self.brack0["text"] = ")"
        self.brack0["command"] = lambda: self.b_pressed(")")
        self.brack0.grid(row = 5, column = 4, sticky = "EW")
        
        self.sqrt = tk.Button(self)
        self.sqrt["text"] = "sqrt"
        self.sqrt["command"] = lambda: self.b_pressed("sqrt(")
        self.sqrt.grid(row = 6, column = 0, sticky = "EW")
        
        self.log = tk.Button(self)
        self.log["text"] = "log"
        self.log["command"] = lambda: self.b_pressed("log(")
        self.log.grid(row = 6, column = 1, sticky = "EW")
        
        self.log10 = tk.Button(self)
        self.log10["text"] = "log10"
        self.log10["command"] = lambda: self.b_pressed("log10(")
        self.log10.grid(row = 6, column = 2, sticky = "EW")
        
        self.copy = tk.Button(self)
        self.copy["text"] = "Copy"
        self.copy["command"] = self.copy
        self.copy.grid(row = 6, column = 3, sticky = "EW")
        
        self.equals = tk.Button(self)
        self.equals["text"] = "="
        self.equals["command"] = self.calculate
        self.equals.grid(row = 6, column = 4, sticky = "EW")
    
    def b_pressed(self, button):
        if self.error.get():
            self.value.set("")
            self.error.set(False)
        
        current_value = self.value.get()
        new_value = current_value + button
        self.value.set(new_value)
        
        # Update pressed button history
        last_pressed = self.b_last_pressed.get()
        self.b_last_pressed.set(";".join([last_pressed, button]))
    
    def function_start(self):
        self.funct.set(True)
        self.value.set("f(x)=")
        last_pressed = self.b_last_pressed.get()
        self.b_last_pressed.set(";".join([last_pressed, "f(x)="]))
        
    def clear(self):
        self.value.set("")
        self.funct.set(False)
        self.error.set(False)
        self.b_last_pressed.set("")
    
    def delete(self):
        current_value = self.value.get()
        last_pressed = self.b_last_pressed.get().split(";")
        
        if current_value.endswith(last_pressed[-1]):
            last_added = last_pressed.pop(-1)
        else:
            last_added = current_value[-1]
        
        new_value = current_value[:-len(last_added)]
        self.value.set(str(new_value))
        self.b_last_pressed.set(";".join(last_pressed))
    
    def copy(self):
        current_value = self.value.get()
        pyperclip.copy(current_value)
        self.clipboard_clear()
        self.clipboard_append(current_value)
    
    def calculate(self):
        if self.funct.get():
            current_value = self.value.get()
            x_vals = linspace(self.x_min.get(), self.x_max.get(),
                              self.n_ticks.get())
            f = current_value.replace("f(x)=", "")
            to_replace = re.findall("\dx", f)
            replacements = [substr.replace("x", "*x") for substr in to_replace]
            for old, new in zip(to_replace, replacements):
                f = f.replace(old, new)
            y_vals = [eval(f) for x in x_vals]
            plt.plot(x_vals, y_vals, "r")
            plt.show()
        else:
            current_value = self.value.get()
            try:
                new_value = eval(current_value)
                self.value.set(new_value)
            except ZeroDivisionError:
                mssg = choice(["Why do you try me?",
                               "Zero isn't much, huh?",
                               "I won't do that."])
                self.value.set(mssg)
                self.error.set(True)
            except:
                mssg = choice(["You f-ed up. Again.",
                               "C'mon man...",
                               "Have you been drinking?",
                               "This is why aliens won't talk to us.",
                               "What do you think you are doing?",
                               "This won't work.",
                               "You're wasting my time!",
                               "Dude!",
                               "Wtf.",
                               "You think this is funny?"
                               ]) if self.error.get() else "Syntax error."
                self.value.set(mssg)
                self.error.set(True)
    
    def cancel(self, x_min_curr, x_max_curr, n_ticks_curr):
        self.x_min.set(x_min_curr)
        self.x_max.set(x_max_curr)
        self.n_ticks.set(n_ticks_curr)
    
    def edit_plot_options(self):
        # Window
        options_window = tk.Toplevel(self)
        options_window.title("Plot options")
        options_window.geometry("100x100")
        
        # Lower plot window limit
        tk.Label(options_window, text = "X-axis min") \
            .grid(row = 0, column = 0)
        x_min_curr = self.x_min.get()
        x_min_opt = tk.Entry(options_window, width = 40)
        x_min_opt["textvariable"] = self.x_min
        x_min_opt.grid(row = 0, column = 1)
        
        # Upper plot window limit
        tk.Label(options_window, text = "X-axis max") \
            .grid(row = 1, column = 0)
        x_max_curr = self.x_max.get()
        x_max_opt = tk.Entry(options_window, width = 40)
        x_max_opt["textvariable"] = self.x_max
        x_max_opt.grid(row = 1, column = 1)
        
        # Upper plot N values
        tk.Label(options_window, text = "X-axis N ticks") \
            .grid(row = 3, column = 0)
        n_ticks_curr = self.n_ticks.get()
        n_opt = tk.Entry(options_window, width = 40)
        n_opt["textvariable"] = self.n_ticks
        n_opt.grid(row = 3, column = 1)
        
        # Plot axis
        tk.Button(options_window, text = "Cancel",
                  command = lambda: [self.cancel(x_min_curr, x_max_curr,
                                                 n_ticks_curr),
                                     options_window.destroy()]) \
            .grid(row = 4, column = 0, stick = "W")
        tk.Button(options_window, text = "Ok",
                  command = lambda: [options_window.destroy()]) \
            .grid(row = 4, column = 1, stick = "W")

#-----------------------------------------------------------------------------|
# Run calculator
root = tk.Tk()
app = Calculator(root)
app.mainloop()
