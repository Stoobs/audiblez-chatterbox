#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# A simple wxWidgets UI for audiblez

import torch.cuda
import numpy as np
import soundfile
import threading
import platform
import subprocess
import io
import os
import wx
import wx.adv  # For DatePickerCtrl, TimePickerCtrl
from wx.lib.agw import flatnotebook as fnb
from wx.lib.agw.ultimatelistctrl import UltimateListCtrl, ULC_REPORT, ULC_SINGLE_SEL
from wx.lib.checkbox import GenCheckBox

from datetime import datetime, time as dt_time  # For schedule dialog
from wx.lib.newevent import NewEvent
from wx.lib.scrolledpanel import ScrolledPanel
from PIL import Image
from tempfile import NamedTemporaryFile
from pathlib import Path
import audiblez.database as db  # Changed import for clarity
import json  # For settings

# from audiblez.database import load_all_user_settings, save_user_setting # Now use db. prefix



# Windows-specific imports for dark mode support
if platform.system() == 'Windows':
    try:
        import ctypes
        from ctypes import wintypes
        WINDOWS_DARK_MODE_AVAILABLE = True
    except ImportError:
        WINDOWS_DARK_MODE_AVAILABLE = False
else:
    WINDOWS_DARK_MODE_AVAILABLE = False

# Theme definitions
palettes = {
    "light": {
        "background": wx.Colour(240, 240, 240),
        "text": wx.Colour(0, 0, 0),
        "text_secondary": wx.Colour(80, 80, 80),
        "panel": wx.Colour(255, 255, 255),
        "border": wx.Colour(200, 200, 200),
        "highlight": wx.Colour(0, 120, 215),
        "highlight_text": wx.Colour(255, 255, 255),
        "button_face": wx.Colour(225, 225, 225),
        "button_text": wx.Colour(0, 0, 0),
        "list_even": wx.Colour(255, 255, 255),
        "list_odd": wx.Colour(245, 245, 245),
        "list_header": wx.Colour(220, 220, 220),
    },
    "dark": {
        "background": wx.Colour(45, 45, 48),
        "text": wx.Colour(230, 230, 230),
        "text_secondary": wx.Colour(180, 180, 180),
        "panel": wx.Colour(60, 60, 63),
        "border": wx.Colour(90, 90, 90),
        "highlight": wx.Colour(90, 156, 248),
        "highlight_text": wx.Colour(255, 255, 255),
        "button_face": wx.Colour(75, 75, 78),
        "button_text": wx.Colour(230, 230, 230),
        "list_even": wx.Colour(60, 60, 63),
        "list_odd": wx.Colour(70, 70, 73),
        "list_header": wx.Colour(80, 80, 83),
    }
}
theme = palettes['light'] # Global theme variable

def get_best_device():
    import torch
    # Only use MPS if available and not on macOS (darwin) or if user has not overridden
    # But, on macOS, check for known problematic libraries and warn or fallback to CPU if needed.
    import platform
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Extra runtime check: try allocating a tensor on MPS to see if it works
        try:
            x = torch.ones(1, device="mps")
            # Optionally: check for known problematic libraries here
            return "mps"
        except Exception:
            # If allocation fails, fallback to CPU
            return "cpu"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# Windows dark mode support functions
def set_windows_dark_mode(hwnd, enable_dark_mode):
    """Enable or disable dark mode for a Windows window handle"""
    if not WINDOWS_DARK_MODE_AVAILABLE:
        return False
    
    try:
        # Windows 10 version 1903 and later
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        value = ctypes.c_int(1 if enable_dark_mode else 0)
        
        # Try to set dark mode attribute
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 
            ctypes.byref(value), ctypes.sizeof(value)
        )
        return True
    except:
        return False

def apply_windows_dark_mode_to_window(window, enable_dark_mode):
    """Apply Windows dark mode to a wxPython window"""
    if not WINDOWS_DARK_MODE_AVAILABLE or not hasattr(window, 'GetHandle'):
        return False
    
    try:
        hwnd = window.GetHandle()
        return set_windows_dark_mode(hwnd, enable_dark_mode)
    except:
        return False

# Custom themed dialog classes
class ThemedMessageDialog(wx.Dialog):
    """Custom themed message dialog to replace wx.MessageBox"""
    
    def __init__(self, parent, message, caption="Message", style=wx.OK):
        super().__init__(parent, title=caption, size=(400, 200))
        self.SetMinSize((300, 150))
        
        # Main panel
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Icon and message area
        content_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Add icon based on style
        if style & wx.ICON_ERROR:
            icon = wx.ArtProvider.GetBitmap(wx.ART_ERROR, wx.ART_MESSAGE_BOX)
        elif style & wx.ICON_WARNING:
            icon = wx.ArtProvider.GetBitmap(wx.ART_WARNING, wx.ART_MESSAGE_BOX)
        elif style & wx.ICON_INFORMATION:
            icon = wx.ArtProvider.GetBitmap(wx.ART_INFORMATION, wx.ART_MESSAGE_BOX)
        elif style & wx.ICON_QUESTION:
            icon = wx.ArtProvider.GetBitmap(wx.ART_QUESTION, wx.ART_MESSAGE_BOX)
        else:
            icon = None
        
        if icon:
            icon_ctrl = wx.StaticBitmap(panel, bitmap=icon)
            content_sizer.Add(icon_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 10)
        
        # Message text
        msg_text = wx.StaticText(panel, label=message)
        msg_text.Wrap(350)
        content_sizer.Add(msg_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 10)
        
        main_sizer.Add(content_sizer, 1, wx.EXPAND | wx.ALL, 10)
        
        # Button area
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        if style & wx.YES_NO:
            yes_btn = wx.Button(panel, wx.ID_YES, "Yes")
            no_btn = wx.Button(panel, wx.ID_NO, "No")
            button_sizer.Add(yes_btn, 0, wx.ALL, 5)
            button_sizer.Add(no_btn, 0, wx.ALL, 5)
            yes_btn.SetDefault()
        elif (hasattr(wx, "OK_CANCEL") and style & wx.OK_CANCEL) or (style & (wx.OK | wx.CANCEL) == (wx.OK | wx.CANCEL)):
            ok_btn = wx.Button(panel, wx.ID_OK, "OK")
            cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
            button_sizer.Add(ok_btn, 0, wx.ALL, 5)
            button_sizer.Add(cancel_btn, 0, wx.ALL, 5)
            ok_btn.SetDefault()
        else:
            ok_btn = wx.Button(panel, wx.ID_OK, "OK")
            button_sizer.Add(ok_btn, 0, wx.ALL, 5)
            ok_btn.SetDefault()
        
        main_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        panel.SetSizer(main_sizer)
        
        # Apply theme
        self.apply_theme()
        
        # Center on parent
        self.CenterOnParent()
        
        # Apply Windows dark mode if available
        if theme == palettes['dark']:
            apply_windows_dark_mode_to_window(self, True)
    
    def apply_theme(self):
        """Apply the current theme to this dialog"""
        self.SetBackgroundColour(theme['background'])
        
        def theme_children(widget):
            if hasattr(widget, 'GetChildren'):
                for child in widget.GetChildren():
                    if isinstance(child, wx.Panel):
                        child.SetBackgroundColour(theme['background'])
                    elif isinstance(child, wx.StaticText):
                        child.SetForegroundColour(theme['text'])
                        child.SetBackgroundColour(theme['background'])
                    elif isinstance(child, wx.Button):
                        child.SetBackgroundColour(theme['button_face'])
                        child.SetForegroundColour(theme['button_text'])
                    
                    theme_children(child)
        
        theme_children(self)
        self.Refresh()

def show_themed_message(parent, message, caption="Message", style=wx.OK):
    """Show a themed message dialog"""
    dialog = ThemedMessageDialog(parent, message, caption, style)
    result = dialog.ShowModal()
    dialog.Destroy()
    return result

class ThemedFileDialog(wx.Dialog):
    """Custom themed file dialog - simplified version"""
    
    def __init__(self, parent, message="Choose a file", defaultDir="", 
                 defaultFile="", wildcard="*.*", style=wx.FD_OPEN):
        super().__init__(parent, title=message, size=(600, 400))
        
        self.selected_path = None
        self.file_style = style
        
        # Create native file dialog but try to theme it
        self.native_dialog = wx.FileDialog(
            self, message=message, defaultDir=defaultDir,
            defaultFile=defaultFile, wildcard=wildcard, style=style
        )
        
        # Apply Windows dark mode if available
        if theme == palettes['dark']:
            apply_windows_dark_mode_to_window(self.native_dialog, True)
        
    def ShowModal(self):
        """Show the native file dialog with theming applied"""
        return self.native_dialog.ShowModal()
        
    def GetPath(self):
        """Get the selected file path"""
        return self.native_dialog.GetPath()
        
    def GetPaths(self):
        """Get multiple selected file paths"""
        return self.native_dialog.GetPaths()
        
    def Destroy(self):
        """Clean up resources"""
        self.native_dialog.Destroy()
        super().Destroy()

def show_themed_dir_dialog(parent, message="Choose a directory", defaultPath="", style=wx.DD_DEFAULT_STYLE):
    """Show a themed directory dialog"""
    dialog = wx.DirDialog(parent, message, defaultPath, style)
    
    # Apply Windows dark mode if available
    if theme == palettes['dark']:
        apply_windows_dark_mode_to_window(dialog, True)
    
    result = dialog.ShowModal()
    path = dialog.GetPath() if result == wx.ID_OK else None
    dialog.Destroy()
    return result, path

def show_themed_file_dialog(parent, message="Choose a file", defaultDir="", 
                           defaultFile="", wildcard="*.*", style=wx.FD_OPEN):
    """Show a themed file dialog"""
    dialog = ThemedFileDialog(parent, message, defaultDir, defaultFile, wildcard, style)
    result = dialog.ShowModal()
    path = dialog.GetPath() if result == wx.ID_OK else None
    dialog.Destroy()
    return result, path

# Enhanced theme palettes with comprehensive color definitions
palettes = {
    'light': {
        "background": wx.Colour(255, 255, 255),
        "text": wx.Colour(0, 0, 0),
        "panel": wx.Colour(240, 240, 240),
        "border": wx.Colour(200, 200, 200),
        "highlight": wx.Colour(0, 120, 215),
        "highlight_text": wx.Colour(255, 255, 255),
        "button_face": wx.Colour(230, 230, 230),
        "button_text": wx.Colour(0, 0, 0),
        "list_even": wx.Colour(255, 255, 255),
        "list_odd": wx.Colour(248, 248, 248),
        "list_header": wx.Colour(220, 220, 220),
    },
    'dark': {
        "background": wx.Colour(32, 32, 32),
        "text": wx.Colour(255, 255, 255),
        "panel": wx.Colour(60, 60, 63),
        "border": wx.Colour(90, 90, 90),
        "highlight": wx.Colour(90, 156, 248),
        "highlight_text": wx.Colour(255, 255, 255),
        "button_face": wx.Colour(75, 75, 78),
        "button_text": wx.Colour(230, 230, 230),
        "list_even": wx.Colour(60, 60, 63),
        "list_odd": wx.Colour(70, 70, 73),
        "list_header": wx.Colour(80, 80, 83),
    }
}

theme = palettes['light'] # Global theme variable

EVENTS = {
    'CORE_STARTED': NewEvent(),
    'CORE_PROGRESS': NewEvent(),
    'CORE_CHAPTER_STARTED': NewEvent(),
    'CORE_CHAPTER_FINISHED': NewEvent(),
    'CORE_FINISHED': NewEvent()
}

border = 5


class CustomGauge(wx.Panel):
    def __init__(self, parent, range_val=100, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.range = range_val
        self.value = 0
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def SetValue(self, value):
        self.value = max(0, min(self.range, value))
        self.Refresh()

    def GetValue(self):
        return self.value

    def SetRange(self, range_val):
        self.range = range_val
        self.Refresh()

    def on_paint(self, event):
        dc = wx.PaintDC(self)
        width, height = self.GetSize()

        # These colors will be replaced by theme colors in Phase 2
        background_color = theme['panel']
        fill_color = theme['highlight']
        border_color = theme['border']

        dc.SetBrush(wx.Brush(background_color))
        dc.SetPen(wx.Pen(border_color))
        dc.DrawRectangle(0, 0, width, height)

        if self.range > 0 and self.value > 0:
            progress_width = int((self.value / self.range) * width)
            dc.SetBrush(wx.Brush(fill_color))
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.DrawRectangle(0, 0, progress_width, height)


class ListBoxComboPopup(wx.ComboPopup):
    def __init__(self, choices=None):
        wx.ComboPopup.__init__(self)
        self.choices = choices if choices is not None else []
        self.listbox = None

    def Create(self, parent):
        self.listbox = wx.ListBox(parent, choices=self.choices)
        self.listbox.Bind(wx.EVT_LISTBOX, self.on_listbox_select)
        return True

    def GetControl(self):
        return self.listbox

    def GetStringValue(self):
        if self.listbox.GetSelection() != wx.NOT_FOUND:
            return self.listbox.GetStringSelection()
        return self.GetComboCtrl().GetValue()

    def on_listbox_select(self, event):
        # This event is needed to select the item and dismiss the popup
        combo = self.GetComboCtrl()
        value = event.GetString()
        combo.SetValue(value)
        self.Dismiss()
        # Manually fire the event to ensure the handler is called
        text_event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED, combo.GetId())
        text_event.SetString(value)
        wx.PostEvent(combo, text_event)

    # The following methods are required by the interface
    def OnPopup(self):
        super().OnPopup()

    def OnDismiss(self):
        super().OnDismiss()


class MainWindow(wx.Frame):
    def _load_epub_file(self, file_path):
        """Helper function to load and process an EPUB file."""
        print(f"Opening file: {file_path}")
        from ebooklib import epub
        from audiblez.core import find_document_chapters_and_extract_texts, find_cover
        from pathlib import Path
        try:
            book = epub.read_epub(file_path)
            meta_title = book.get_metadata('DC', 'title')
            title = meta_title[0][0] if meta_title else Path(file_path).stem
            meta_creator = book.get_metadata('DC', 'creator')
            author = meta_creator[0][0] if meta_creator else 'Unknown Author'
            document_chapters = find_document_chapters_and_extract_texts(book)
            cover = find_cover(book)
            cover_info = {'type': 'epub_cover', 'content': cover.content} if cover else None
            wx.CallAfter(self._load_book_data_into_ui,
                book_title=title,
                book_author=author,
                document_chapters=document_chapters,
                source_path=file_path,
                book_object=book,
                cover_info=cover_info
            )
        except Exception as e:
            print(f"Error opening EPUB file {file_path}: {e}")
            show_themed_message(self, f"Failed to open or parse the EPUB file:\n\n{e}", "EPUB Error", wx.OK | wx.ICON_ERROR)
    def __init__(self, parent, title):
        import wx
        screen_width, screen_h = wx.GetDisplaySize()
        self.window_width = int(screen_width * 0.6)
        super().__init__(parent, title=title, size=(self.window_width, self.window_width * 3 // 4))
        self.theme_name = 'light'
        self.chapters_panel = None
        self.preview_threads = []
        self.selected_chapter = None
        self.selected_book = None
        self.synthesis_in_progress = False

        self.Bind(EVENTS['CORE_STARTED'][1], self.on_core_started)
        self.Bind(EVENTS['CORE_CHAPTER_STARTED'][1], self.on_core_chapter_started)
        self.Bind(EVENTS['CORE_CHAPTER_FINISHED'][1], self.on_core_chapter_finished)
        self.Bind(EVENTS['CORE_PROGRESS'][1], self.on_core_progress)
        self.Bind(EVENTS['CORE_FINISHED'][1], self.on_core_finished)

        self.create_menu()
        self.create_layout()

        # Load user settings
        self.user_settings = db.load_all_user_settings() # Use db prefix
        if not self.user_settings: # Ensure it's a dict
            self.user_settings = {}

        # Restore window size
        geometry = self.user_settings.get('window_geometry')
        if geometry:
            try:
                width, height = map(int, geometry.split('x'))
                self.SetSize((width, height))
            except (ValueError, TypeError):
                pass # Ignore invalid geometry string

        self.Bind(wx.EVT_SIZE, self.on_resize)

        # Apply theme on startup - defer until after UI is fully created
        saved_theme = self.user_settings.get('dark_mode', 'light')
        self.theme_name = saved_theme
        
        # Store the theme to apply later
        self._pending_theme = saved_theme

        # Initialize core attributes that will be set by UI controls,
        # potentially using loaded settings or defaults.
        # These will be properly set in create_params_panel and create_synthesis_panel
        self.custom_rate = None # Default custom rate
        self.m4b_assembly_method = 'original' # Default M4B assembly method

        self.queue_processing_active = False
        self.current_queue_item_index = -1 # To track which item in self.queue_items is being processed
        self.run_queue_button = None # To enable/disable run queue button
        self.schedule_queue_button = None # For scheduling
        self.scheduled_time_text = None # To display scheduled time

        # Load queue from database on startup
        self.queue_items = db.get_queued_items()

        self.Centre()
        self.Show(True)

        # Ensure notebook and tabs are created, then refresh them.
        self.create_notebook_and_tabs()
        self._create_static_panels() # Create the right-hand side panels
        
        # Apply theme after UI is fully constructed
        wx.CallAfter(self._apply_initial_theme)
        wx.CallAfter(self._initial_ui_refresh) # Refresh tabs after UI is fully up

        # Initialize and start schedule checker
        self.schedule_check_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_check_schedule_timer, self.schedule_check_timer)
        self.start_schedule_check_timer()

        # Bind close event to stop timer
        self.Bind(wx.EVT_CLOSE, self.on_close_window)

        default_epub_path = Path('../epub/lewis.epub')
        if default_epub_path.exists():
            wx.CallAfter(self._load_epub_file, str(default_epub_path))

    def on_resize(self, event):
        width, height = self.GetSize()
        db.save_user_setting('window_geometry', f'{width}x{height}')
        event.Skip()

    def on_close_window(self, event):
        if self.schedule_check_timer.IsRunning():
            self.schedule_check_timer.Stop()
        # Add any other cleanup needed before closing
        self.Destroy() # Proceed with closing


    def _apply_initial_theme(self):
        """Apply the saved theme after UI is fully constructed"""
        if hasattr(self, '_pending_theme'):
            # Update dark mode toggle to match saved setting
            if hasattr(self, 'dark_mode_toggle'):
                self.dark_mode_toggle.SetValue(self._pending_theme == 'dark')
            
            # Apply the theme
            self.apply_theme(self._pending_theme)

    def _initial_ui_refresh(self):
        if not hasattr(self, 'notebook'):
            print("Notebook not created, cannot perform initial UI refresh.")
            return
        self.refresh_queue_tab()
        self.refresh_staging_tab()


    def create_notebook_and_tabs(self):
        if hasattr(self, 'notebook') and self.notebook:
            # Notebook and basic tabs might have been created by a previous call or open_epub
            # Ensure sizers exist if we are re-entering or setting up lazily
            if not hasattr(self, 'queue_tab_sizer') and hasattr(self, 'queue_tab_panel') and self.queue_tab_panel:
                self.queue_tab_sizer = wx.BoxSizer(wx.VERTICAL)
                self.queue_tab_panel.SetSizer(self.queue_tab_sizer)
            if not hasattr(self, 'staging_tab_sizer') and hasattr(self, 'staging_tab_panel') and self.staging_tab_panel:
                self.staging_tab_sizer = wx.BoxSizer(wx.VERTICAL)
                self.staging_tab_panel.SetSizer(self.staging_tab_sizer)
            return

        if not hasattr(self, 'splitter_left') or not self.splitter_left:
            if not hasattr(self, 'splitter') or not self.splitter:
                 print("Error: Main splitter panel does not exist. Cannot create notebook.")
                 return
            self.splitter_left = wx.Panel(self.splitter, -1)
            self.left_sizer = wx.BoxSizer(wx.VERTICAL)
            self.splitter_left.SetSizer(self.left_sizer)
            # Add splitter_left to the main splitter_sizer
            self.splitter_sizer.Add(self.splitter_left, 1, wx.ALL | wx.EXPAND, 5)

        self.notebook = fnb.FlatNotebook(self.splitter_left)

        # Chapters Tab
        self.chapters_tab_page = wx.Panel(self.notebook)
        self.notebook.AddPage(self.chapters_tab_page, "Chapters")
        chapters_page_sizer = wx.BoxSizer(wx.VERTICAL) # Create sizer for chapters page
        self.chapters_tab_page.SetSizer(chapters_page_sizer) # Set sizer
        if not hasattr(self, 'chapters_panel'): # If open_epub hasn't run
            placeholder_text = wx.StaticText(self.chapters_tab_page, label="Open an EPUB file to see chapters here.")
            chapters_page_sizer.Add(placeholder_text, 0, wx.ALL | wx.ALIGN_CENTER, 15)
            self.chapters_tab_page.Layout()

        # Staging Tab Panel
        self.staging_tab_panel = ScrolledPanel(self.notebook, -1, style=wx.TAB_TRAVERSAL | wx.SUNKEN_BORDER)
        self.staging_tab_sizer = wx.BoxSizer(wx.VERTICAL)
        self.staging_tab_panel.SetSizer(self.staging_tab_sizer)
        self.notebook.AddPage(self.staging_tab_panel, "Staging")
        if not self.staging_tab_sizer.GetChildren():
            placeholder_staging = wx.StaticText(self.staging_tab_panel, label="Staged books will appear here.")
            self.staging_tab_sizer.Add(placeholder_staging, 0, wx.ALL | wx.ALIGN_CENTER, 15)
            self.staging_tab_panel.Layout()
            self.staging_tab_panel.SetupScrolling()

        # Queue Tab Panel
        self.queue_tab_panel = ScrolledPanel(self.notebook, -1, style=wx.TAB_TRAVERSAL | wx.SUNKEN_BORDER)
        self.queue_tab_sizer = wx.BoxSizer(wx.VERTICAL)
        self.queue_tab_panel.SetSizer(self.queue_tab_sizer)
        self.notebook.AddPage(self.queue_tab_panel, "Queue")
        if not self.queue_tab_sizer.GetChildren():
            # print("DEBUG: Creating initial placeholder for Queue tab.")
            placeholder_queue = wx.StaticText(self.queue_tab_panel, label="Queued items will appear here.")
            self.queue_tab_sizer.Add(placeholder_queue, 0, wx.ALL | wx.ALIGN_CENTER, 15)
            self.queue_tab_panel.Layout()
            self.queue_tab_panel.SetupScrolling()

        self.left_sizer.Add(self.notebook, 1, wx.ALL | wx.EXPAND, 5)

        if hasattr(self.splitter_left, 'Layout'): self.splitter_left.Layout()
        if hasattr(self.splitter, 'Layout'): self.splitter.Layout()
        self.Layout()


    def create_menu(self):
        menubar = wx.MenuBar()
        file_menu = wx.Menu()
        open_item = wx.MenuItem(file_menu, wx.ID_OPEN, "&Open\tCtrl+O")
        file_menu.Append(open_item)
        self.Bind(wx.EVT_MENU, self.on_open, open_item)  # Bind the event

        exit_item = wx.MenuItem(file_menu, wx.ID_EXIT, "&Exit\tCtrl+Q")
        file_menu.Append(exit_item)
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)

        menubar.Append(file_menu, "&File")
        self.SetMenuBar(menubar)
        
        # Store reference for theming
        self.menubar = menubar
        self.file_menu = file_menu

    def on_core_started(self, event):
        print('CORE_STARTED')
        self.progress_bar_label.Show()
        self.progress_bar.Show()
        self.progress_bar.SetValue(0)
        self.progress_bar.Layout()
        self.eta_label.Show()
        self.params_panel.Layout()
        self.synth_panel.Layout()

    def on_core_chapter_started(self, event):
        # print('CORE_CHAPTER_STARTED', event.chapter_index)
        self.set_table_chapter_status(event.chapter_index, "⏳ In Progress")

    def on_core_chapter_finished(self, event):
        # print('CORE_CHAPTER_FINISHED', event.chapter_index)
        self.set_table_chapter_status(event.chapter_index, "✅ Done")
        self.start_button.Show()

    def on_core_progress(self, event):
        # print('CORE_PROGRESS', event.progress)
        self.progress_bar.SetValue(event.stats.progress)
        self.progress_bar_label.SetLabel(f"Synthesis Progress: {event.stats.progress}%")
        self.eta_label.SetLabel(f"Estimated Time Remaining: {event.stats.eta}")
        self.synth_panel.Layout()

    def on_core_finished(self, event):
        # Ensure progress bar shows 100% on completion
        self.progress_bar.SetValue(100)
        self.progress_bar_label.SetLabel("Synthesis Progress: 100%")
        self.synth_panel.Layout() # Refresh layout to show update immediately

        self.synthesis_in_progress = False # This is for single book synthesis
        # For queue, self.queue_processing_active is used.

        # If queue was active, on_core_finished handles the next item or cleanup
        if self.queue_processing_active:
            if self.current_queue_item_index < len(self.queue_items):
                 # Update status of the completed item
                item_data = self.queue_items[self.current_queue_item_index] # This is a reference from self.queue_items
                # Ensure 'id' exists, as it's crucial for DB updates.
                if 'id' in item_data:
                    # Check if CoreThread passed an error status
                    if hasattr(event, 'error_message') and event.error_message:
                        db.update_queue_item_status(item_data['id'], 'error')
                        item_data['status'] = f"⚠️ Error ({event.error_message})"
                        print(f"Error processing queue item {item_data['id']}: {event.error_message}")
                    else:
                        db.update_queue_item_status(item_data['id'], 'completed')
                        item_data['status'] = "✅ Completed"

                        # If completed, update status of staged chapters in DB and refresh staging UI
                        processed_staged_chapter_ids = []
                        for chap_info in item_data.get('chapters', []):
                            staged_chapter_id = chap_info.get('staged_chapter_id')
                            if staged_chapter_id:
                                db.update_staged_chapter_status_in_db(staged_chapter_id, 'completed')
                                processed_staged_chapter_ids.append(staged_chapter_id)
                        if processed_staged_chapter_ids:
                            self.update_staging_tab_for_processed_chapters(processed_staged_chapter_ids)
                else:
                    print(f"Error: Queue item {item_data.get('book_title')} missing 'id', cannot update DB status.")

            # Try to process the next item
            self.current_queue_item_index += 1
            self.process_next_queue_item() # This method will handle actual processing
        else:
            # This was a single synthesis, not from queue
            # self.open_folder_with_explorer(self.output_folder_text_ctrl.GetValue())
            # Re-enable start button and params if it was a single synthesis
            self.start_button.Enable()
            self.params_panel.Enable()
            if hasattr(self, 'table'): self.table.Enable(True)


    def create_layout(self):
        # Panels layout looks like this:
        # splitter
        #     splitter_left
        #         chapters_panel
        #     splitter_right
        #         center_panel
        #             text_area
        #         right_panel
        #             book_info_panel_box
        #                 book_info_panel
        #                     cover_bitmap
        #                     book_details_panel
        #             param_panel_box
        #                  param_panel
        #                      ...
        #             synth_panel_box
        #                  synth_panel
        #                      start_button
        #                      ...

        top_panel = wx.Panel(self)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_panel.SetSizer(top_sizer)

        # Open Epub button
        open_epub_button = wx.Button(top_panel, label="📁 Open EPUB")
        open_epub_button.Bind(wx.EVT_BUTTON, self.on_open)
        top_sizer.Add(open_epub_button, 0, wx.ALL, 5)

        # Open with Calibre button
        open_calibre_button = wx.Button(top_panel, label="📖 Open with Calibre")
        open_calibre_button.Bind(wx.EVT_BUTTON, self.on_open_with_calibre)
        top_sizer.Add(open_calibre_button, 0, wx.ALL, 5)

        # Open Markdown .md
        # open_md_button = wx.Button(top_panel, label="📁 Open Markdown (.md)")
        # open_md_button.Bind(wx.EVT_BUTTON, self.on_open)
        # top_sizer.Add(open_md_button, 0, wx.ALL, 5)

        # Open .txt
        # open_txt_button = wx.Button(top_panel, label="📁 Open .txt")
        # open_txt_button.Bind(wx.EVT_BUTTON, self.on_open)
        # top_sizer.Add(open_txt_button, 0, wx.ALL, 5)

        # Open PDF
        # open_pdf_button = wx.Button(top_panel, label="📁 Open PDF")
        # open_pdf_button.Bind(wx.EVT_BUTTON, self.on_open)
        # top_sizer.Add(open_pdf_button, 0, wx.ALL, 5)

        # About button
        help_button = wx.Button(top_panel, label="ℹ️ About")
        help_button.Bind(wx.EVT_BUTTON, lambda event: self.about_dialog())
        top_sizer.Add(help_button, 0, wx.ALL, 5)

        # Dark Mode Toggle
        self.dark_mode_toggle = GenCheckBox(top_panel, label="🌙 Dark Mode")
        self.dark_mode_toggle.Bind(wx.EVT_CHECKBOX, self.on_toggle_dark_mode)
        top_sizer.Add(self.dark_mode_toggle, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)

        # self.splitter = wx.SplitterWindow(self, -1)
        # self.splitter.SetSashGravity(0.9)
        self.splitter = wx.Panel(self)
        self.splitter_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.splitter.SetSizer(self.splitter_sizer)

        self.main_sizer.Add(top_panel, 0, wx.ALL | wx.EXPAND, 5)
        self.main_sizer.Add(self.splitter, 1, wx.EXPAND) # self.splitter is a Panel that will be split by self.splitter_sizer

        # The main content panels (notebook, chapter lists, etc.) are created
        # in create_notebook_and_tabs() and _create_static_panels() during startup.

    def _create_static_panels(self):
        # This function creates the main UI panels that are always present.
        # It's called once at startup.
        self.splitter_right = wx.Panel(self.splitter)
        self.splitter_sizer.Add(self.splitter_right, 2, wx.ALL | wx.EXPAND, 5)

        # Create Center Panel (for text preview)
        self.center_panel = wx.Panel(self.splitter_right)
        self.center_sizer = wx.BoxSizer(wx.VERTICAL)
        self.center_panel.SetSizer(self.center_sizer)
        self.text_area = wx.TextCtrl(self.center_panel, style=wx.TE_MULTILINE, size=(int(self.window_width * 0.4), -1))
        font = wx.Font(14, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.text_area.SetFont(font)
        self.text_area.Bind(wx.EVT_TEXT, lambda event: setattr(self.selected_chapter, 'extracted_text', self.text_area.GetValue()) if self.selected_chapter else None)
        self.chapter_label = wx.StaticText(self.center_panel, label="No chapter selected.")
        
        # Create a horizontal sizer for buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        preview_button = wx.Button(self.center_panel, label="🔊 Preview")
        preview_button.Bind(wx.EVT_BUTTON, self.on_preview_chapter)
        button_sizer.Add(preview_button, 0, wx.ALL, 5)
        
        tidy_button = wx.Button(self.center_panel, label="🧹 Tidy Text")
        tidy_button.Bind(wx.EVT_BUTTON, self.on_tidy_text)
        tidy_button.SetToolTip("Clean up excessive line breaks and improve text formatting")
        button_sizer.Add(tidy_button, 0, wx.ALL, 5)
        
        self.center_sizer.Add(self.chapter_label, 0, wx.ALL, 5)
        self.center_sizer.Add(button_sizer, 0, wx.ALL, 5)
        self.center_sizer.Add(self.text_area, 1, wx.ALL | wx.EXPAND, 5)

        # Create Right Panel (for details, params, synth)
        self.create_right_panel(self.splitter_right)

        splitter_right_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.splitter_right.SetSizer(splitter_right_sizer)
        splitter_right_sizer.Add(self.center_panel, 1, wx.ALL | wx.EXPAND, 5)
        splitter_right_sizer.Add(self.right_panel, 1, wx.ALL | wx.EXPAND, 5)

        # Initially, these panels might be disabled until a book is loaded.
        self.splitter_right.Disable()

    def on_toggle_dark_mode(self, event):
        self.theme_name = 'dark' if event.IsChecked() else 'light'
        db.save_user_setting('dark_mode', self.theme_name)
        self.apply_theme(self.theme_name)

    def apply_theme(self, theme_name):
        global theme
        theme = palettes[theme_name]
        self.theme_name = theme_name

        # Apply Windows dark mode if available
        if theme_name == 'dark':
            apply_windows_dark_mode_to_window(self, True)
        else:
            apply_windows_dark_mode_to_window(self, False)

        # Update the toggle state without firing event
        is_dark = (theme_name == 'dark')
        if hasattr(self, 'dark_mode_toggle'):
            # Block the event to prevent recursion
            self.dark_mode_toggle.Unbind(wx.EVT_CHECKBOX)
            self.dark_mode_toggle.SetValue(is_dark)
            self.dark_mode_toggle.Bind(wx.EVT_CHECKBOX, self.on_toggle_dark_mode)

        # --- Enhanced recursive function to apply theme to all controls ---
        def apply_to_children(parent_widget):
            if not hasattr(parent_widget, 'GetChildren'): 
                return
                
            for child in parent_widget.GetChildren():
                if not child: 
                    continue

                try:
                    # Force refresh on all widgets
                    child.SetBackgroundColour(theme['background'])
                    child.SetForegroundColour(theme['text'])
                    
                    # Specific widget handling
                    if isinstance(child, wx.StaticText):
                        child.SetForegroundColour(theme['text'])
                        child.SetBackgroundColour(theme['background'])
                    
                    elif isinstance(child, (wx.Button, wx.ToggleButton)):
                        child.SetBackgroundColour(theme['button_face'])
                        child.SetForegroundColour(theme['button_text'])
                    
                    elif isinstance(child, wx.TextCtrl):
                        child.SetBackgroundColour(theme['panel'])
                        child.SetForegroundColour(theme['text'])
                    
                    elif isinstance(child, GenCheckBox):
                        child.SetBackgroundColour(theme['background'])
                        child.SetForegroundColour(theme['text'])
                    
                    elif isinstance(child, (wx.SpinCtrl, wx.SpinCtrlDouble)):
                        child.SetBackgroundColour(theme['panel'])
                        child.SetForegroundColour(theme['text'])
                    
                    elif isinstance(child, (wx.Panel, ScrolledPanel)):
                        child.SetBackgroundColour(theme['background'])
                        child.SetForegroundColour(theme['text'])
                    
                    elif hasattr(child, 'SetTabAreaColour'):  # FlatNotebook
                        child.SetBackgroundColour(theme['background'])
                        child.SetTabAreaColour(theme['panel'])
                        child.SetActiveTabColour(theme['highlight'])
                        child.SetNonActiveTabTextColour(theme['text'])
                        child.SetActiveTabTextColour(theme['highlight_text'])
                    
                    # Force refresh the widget
                    child.Refresh()
                    
                    # Recurse to children
                    apply_to_children(child)
                        
                except Exception as e:
                    # Continue if theming fails for a specific control
                    print(f"Theme error on {type(child).__name__}: {e}")

        # --- Main Theme Application ---
        # Set the main window colors first
        self.SetBackgroundColour(theme['background'])
        self.SetForegroundColour(theme['text'])
        
        # Apply to all children recursively
        apply_to_children(self)

        # --- Style Specific Complex Widgets ---
        if hasattr(self, 'notebook') and self.notebook:
            self.notebook.SetBackgroundColour(theme['background'])
            if hasattr(self.notebook, 'SetTabAreaColour'):
                self.notebook.SetTabAreaColour(theme['panel'])
                self.notebook.SetActiveTabColour(theme['highlight'])
                self.notebook.SetNonActiveTabTextColour(theme['text'])
                self.notebook.SetActiveTabTextColour(theme['highlight_text'])

        # Style specific known controls
        if hasattr(self, 'table') and self.table:
            self.table.SetBackgroundColour(theme['panel'])
            self.table.SetForegroundColour(theme['text'])
            for i in range(self.table.GetItemCount()):
                if i % 2 == 0:
                    self.table.SetItemBackgroundColour(i, theme['list_even'])
                else:
                    self.table.SetItemBackgroundColour(i, theme['list_odd'])
                self.table.SetItemTextColour(i, theme['text'])

        # Force refresh everything
        self.Refresh()
        self.Update()
        self.Layout()
        
        # Refresh all child windows
        for child in self.GetChildren():
            if child:
                child.Refresh()
                child.Update()


    def about_dialog(self):
        msg = ("A simple tool to generate audiobooks from EPUB files using Chatterbox-TTS voice models.\n" +
               "Distributed under the MIT License.\n\n" +
               "by Claudio Santini 2025\nand many contributors.\n\n" +
               "https://claudio.uk\n\n")
        show_themed_message(self, msg, "Audiblez", wx.OK | wx.ICON_INFORMATION)

    def create_right_panel(self, splitter_right):
        self.right_panel = wx.Panel(splitter_right)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_panel.SetSizer(self.right_sizer)

        # --- Replacement for StaticBoxSizer ---
        book_details_container = wx.Panel(self.right_panel, style=wx.BORDER_THEME)
        container_sizer = wx.BoxSizer(wx.VERTICAL)
        book_details_container.SetSizer(container_sizer)

        label = wx.StaticText(book_details_container, label="Book Details")
        font = label.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        label.SetFont(font)
        container_sizer.Add(label, 0, wx.ALL & ~wx.BOTTOM, 5)

        self.book_info_panel = wx.Panel(book_details_container, style=wx.BORDER_NONE)
        self.book_info_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.book_info_panel.SetSizer(self.book_info_sizer)
        container_sizer.Add(self.book_info_panel, 1, wx.ALL | wx.EXPAND, 5)
        self.right_sizer.Add(book_details_container, 1, wx.ALL | wx.EXPAND, 5)

        # Add cover image
        self.cover_bitmap = wx.StaticBitmap(self.book_info_panel, -1)
        self.book_info_sizer.Add(self.cover_bitmap, 0, wx.ALL, 5)

        self.cover_bitmap.Refresh()
        self.book_info_panel.Refresh()
        self.book_info_panel.Layout()
        self.cover_bitmap.Layout()

        self.create_book_details_panel()
        self.create_params_panel()
        self.create_synthesis_panel()

    def create_book_details_panel(self):
        book_details_panel = wx.Panel(self.book_info_panel)
        book_details_sizer = wx.GridBagSizer(10, 10)
        book_details_panel.SetSizer(book_details_sizer)
        self.book_info_sizer.Add(book_details_panel, 1, wx.ALL | wx.EXPAND, 5)

        # Add title
        title_label = wx.StaticText(book_details_panel, label="Title:")
        self.title_text = wx.StaticText(book_details_panel, label="")
        title_text = self.title_text
        book_details_sizer.Add(title_label, pos=(0, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(title_text, pos=(0, 1), flag=wx.ALL, border=5)

        # Add Author
        author_label = wx.StaticText(book_details_panel, label="Author:")
        self.author_text = wx.StaticText(book_details_panel, label="")
        author_text = self.author_text
        book_details_sizer.Add(author_label, pos=(1, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(author_text, pos=(1, 1), flag=wx.ALL, border=5)

        # Add Total length
        length_label = wx.StaticText(book_details_panel, label="Total Length:")
        self.length_text = wx.StaticText(book_details_panel, label="")
        length_text = self.length_text
        book_details_sizer.Add(length_label, pos=(2, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(length_text, pos=(2, 1), flag=wx.ALL, border=5)

    def create_params_panel(self):
        # --- Replacement for StaticBoxSizer ---
        panel_container = wx.Panel(self.right_panel, style=wx.BORDER_THEME)
        container_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_container.SetSizer(container_sizer)

        label = wx.StaticText(panel_container, label="Audiobook Parameters")
        font = label.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        label.SetFont(font)
        container_sizer.Add(label, 0, wx.ALL & ~wx.BOTTOM, 5)

        panel = self.params_panel = wx.Panel(panel_container)
        container_sizer.Add(panel, 1, wx.ALL | wx.EXPAND, 5)
        self.right_sizer.Add(panel_container, 1, wx.ALL | wx.EXPAND, 5)
        sizer = wx.GridBagSizer(10, 10)
        panel.SetSizer(sizer)

        engine_label = wx.StaticText(panel, label="Engine:")
        engine_toggle_panel = wx.Panel(panel)
        engine_toggle_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        engine_toggle_panel.SetSizer(engine_toggle_panel_sizer)

        # Always add CPU toggle
        self.cpu_toggle = wx.ToggleButton(engine_toggle_panel, label="CPU")
        engine_toggle_panel_sizer.Add(self.cpu_toggle, 0, wx.ALL, 5)
        self.engine_toggles = [self.cpu_toggle]

        # Add CUDA toggle if available
        self.cuda_toggle = None
        if torch.cuda.is_available():
            self.cuda_toggle = wx.ToggleButton(engine_toggle_panel, label="CUDA")
            engine_toggle_panel_sizer.Add(self.cuda_toggle, 0, wx.ALL, 5)
            self.engine_toggles.append(self.cuda_toggle)

        # Add MPS toggle if available (macOS)
        self.mps_toggle = None
        try:
            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception as e:
            mps_available = False
        if mps_available:
            self.mps_toggle = wx.ToggleButton(engine_toggle_panel, label="MPS")
            engine_toggle_panel_sizer.Add(self.mps_toggle, 0, wx.ALL, 5)
            self.engine_toggles.append(self.mps_toggle)

        def on_select_engine(engine_type):
            torch.set_default_device(engine_type)
            db.save_user_setting('engine', engine_type)
            print(f"Engine set to {engine_type} and saved.")

        def on_engine_toggle(event):
            toggled_button = event.GetEventObject()
            toggled_button.SetValue(True)
            for toggle in self.engine_toggles:
                if toggle != toggled_button:
                    toggle.SetValue(False)
            if toggled_button == self.cpu_toggle:
                engine_type = 'cpu'
            elif self.cuda_toggle and toggled_button == self.cuda_toggle:
                engine_type = 'cuda'
            elif self.mps_toggle and toggled_button == self.mps_toggle:
                engine_type = 'mps'
            else:
                engine_type = 'cpu'
            on_select_engine(engine_type)

        self.cpu_toggle.Bind(wx.EVT_TOGGLEBUTTON, on_engine_toggle)
        if self.cuda_toggle:
            self.cuda_toggle.Bind(wx.EVT_TOGGLEBUTTON, on_engine_toggle)
        if self.mps_toggle:
            self.mps_toggle.Bind(wx.EVT_TOGGLEBUTTON, on_engine_toggle)

        # Load saved engine or set default
        saved_engine = self.user_settings.get('engine')
        device = get_best_device()
        if saved_engine in ['cuda', 'mps', 'cpu']:
            if saved_engine == 'cuda' and torch.cuda.is_available():
                device = 'cuda'
            elif saved_engine == 'mps' and self.mps_toggle:
                device = 'mps'
            elif saved_engine == 'cpu':
                device = 'cpu'

        # Set toggle state
        if device == 'cuda' and self.cuda_toggle:
            self.cuda_toggle.SetValue(True)
        elif device == 'mps' and self.mps_toggle:
            self.mps_toggle.SetValue(True)
        else:
            self.cpu_toggle.SetValue(True)

        torch.set_default_device(device)

        sizer.Add(engine_label, pos=(0, 0), flag=wx.ALL, border=5)
        sizer.Add(engine_toggle_panel, pos=(0, 1), flag=wx.ALL, border=5)

        # Add file dialog selector to select output folder
        output_folder_label = wx.StaticText(panel, label="Output Folder:")
        self.output_folder_text_ctrl = wx.TextCtrl(panel, value=os.path.abspath('.'))
        self.output_folder_text_ctrl.SetEditable(False)
        output_folder_button = wx.Button(panel, label="📂 Select")
        output_folder_button.Bind(wx.EVT_BUTTON, self.open_output_folder_dialog)
        sizer.Add(output_folder_label, pos=(3, 0), flag=wx.ALL, border=5)
        sizer.Add(self.output_folder_text_ctrl, pos=(3, 1), flag=wx.ALL | wx.EXPAND, border=5)
        sizer.Add(output_folder_button, pos=(4, 1), flag=wx.ALL, border=5)

        # --- Chatterbox Model Parameters ---
        chatterbox_label = wx.StaticText(panel, label="Chatterbox Model Parameters:")
        font2 = chatterbox_label.GetFont()
        font2.SetWeight(wx.FONTWEIGHT_BOLD)
        chatterbox_label.SetFont(font2)
        sizer.Add(chatterbox_label, pos=(5, 0), flag=wx.ALL, border=5)

        # Voice Style Presets
        presets_label = wx.StaticText(panel, label="Voice Style Presets:")
        self.presets_choice = wx.Choice(panel, choices=[
            "Custom", "Neutral", "Calm", "Expressive", "Dramatic", "Storytelling", "Audiobook", "Fast Speaker"
        ])
        self.presets_choice.SetSelection(0)  # Default to "Custom"
        self.presets_choice.Bind(wx.EVT_CHOICE, self.on_preset_selected)
        sizer.Add(presets_label, pos=(5, 1), flag=wx.ALL, border=5)
        sizer.Add(self.presets_choice, pos=(6, 1), flag=wx.ALL, border=5)

        self.model_config_controls = {}
        # Add sliders/text for each parameter
        param_defs = [
            ("exaggeration", 0.0, 2.0, 0.6, 0.01),
            ("temperature", 0.0, 2.0, 0.8, 0.01),
            ("cfg_weight", 0.0, 10.0, 0.7, 0.01),
            ("min_p", 0.0, 1.0, 0.05, 0.01),
            ("top_p", 0.0, 1.0, 0.95, 0.01),
            ("repetition_penalty", 0.0, 5.0, 1.0, 0.01),
        ]
        for idx, (param, minv, maxv, default, step) in enumerate(param_defs):
            label = wx.StaticText(panel, label=param.replace('_', ' ').capitalize() + ":")
            ctrl = wx.SpinCtrlDouble(panel, min=minv, max=maxv, initial=default, inc=step)
            # Bind change events to update preset to "Custom" when manually changed
            ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_parameter_changed)
            self.model_config_controls[param] = ctrl
            sizer.Add(label, pos=(7+idx, 0), flag=wx.ALL, border=5)
            sizer.Add(ctrl, pos=(7+idx, 1), flag=wx.ALL, border=5)

        # --- Voice Clone Sample File Picker ---
        clone_label = wx.StaticText(panel, label="Voice Clone Sample:")
        # Load saved voice clone sample path
        saved_voice_clone_sample = self.user_settings.get('voice_clone_sample', '') or ''
        self.clone_sample_path = wx.TextCtrl(panel, value=saved_voice_clone_sample, style=wx.TE_READONLY)
        clone_button = wx.Button(panel, label="Select Sample File")
        def on_select_clone_sample(event):
            result, path = show_themed_file_dialog(self, "Select Voice Clone Sample", 
                                                 wildcard="Audio files (*.wav;*.mp3;*.m4a)|*.wav;*.mp3;*.m4a|All files (*.*)|*.*", 
                                                 style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
            if result == wx.ID_OK and path:
                self.clone_sample_path.SetValue(path)
                # Save the selected voice clone sample to database
                db.save_user_setting('voice_clone_sample', path)
        clone_button.Bind(wx.EVT_BUTTON, on_select_clone_sample)
        sizer.Add(clone_label, pos=(13, 0), flag=wx.ALL, border=5)
        sizer.Add(self.clone_sample_path, pos=(13, 1), flag=wx.ALL | wx.EXPAND, border=5)
        sizer.Add(clone_button, pos=(14, 1), flag=wx.ALL, border=5)

        # M4B Assembly Method
        m4b_assembly_label = wx.StaticText(panel, label="M4B Assembly:")
        m4b_assembly_panel = wx.Panel(panel)
        m4b_assembly_sizer = wx.BoxSizer(wx.HORIZONTAL)
        m4b_assembly_panel.SetSizer(m4b_assembly_sizer)

        self.m4b_assembly_toggle = wx.ToggleButton(m4b_assembly_panel, label="Original")
        self.m4b_assembly_method = 'original'
        m4b_assembly_sizer.Add(self.m4b_assembly_toggle, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        sizer.Add(m4b_assembly_label, pos=(15, 0), flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        sizer.Add(m4b_assembly_panel, pos=(15, 1), flag=wx.ALL, border=5)

    def on_preset_selected(self, event):
        """Handle preset selection from dropdown"""
        presets = {
            "neutral": {"exaggeration": 0.5, "cfg_weight": 0.5},
            "calm": {"exaggeration": 0.3, "cfg_weight": 0.6},
            "expressive": {"exaggeration": 0.7, "cfg_weight": 0.4},
            "dramatic": {"exaggeration": 1.0, "cfg_weight": 0.3},
            "storytelling": {"exaggeration": 0.8, "cfg_weight": 0.4},
            "audiobook": {"exaggeration": 0.4, "cfg_weight": 0.6},
            "fast_speaker": {"exaggeration": 0.6, "cfg_weight": 0.3},
        }
        
        selected_preset = event.GetString().lower()
        
        # Skip if Custom is selected
        if selected_preset == "custom":
            return
            
        # Apply preset values if they exist
        if selected_preset in presets:
            preset_values = presets[selected_preset]
            
            # Update the controls with preset values
            for param_name, value in preset_values.items():
                if param_name in self.model_config_controls:
                    self.model_config_controls[param_name].SetValue(value)

    def on_parameter_changed(self, event):
        """Handle manual parameter changes - set preset to Custom"""
        # Set the preset dropdown to "Custom" when user manually changes parameters
        self.presets_choice.SetSelection(0)  # "Custom" is at index 0
        event.Skip()

    def create_synthesis_panel(self):
        # Think and identify layout issue with the folling code
        # --- Replacement for StaticBoxSizer ---
        panel_container = wx.Panel(self.right_panel, style=wx.BORDER_THEME)
        container_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_container.SetSizer(container_sizer)

        label = wx.StaticText(panel_container, label="Audiobook Generation Status")
        font = label.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        label.SetFont(font)
        container_sizer.Add(label, 0, wx.ALL & ~wx.BOTTOM, 5)

        panel = self.synth_panel = wx.Panel(panel_container)
        container_sizer.Add(panel, 1, wx.ALL | wx.EXPAND, 5)
        self.right_sizer.Add(panel_container, 1, wx.ALL | wx.EXPAND, 5)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)

        # Add Start button
        self.start_button = wx.Button(panel, label="🚀 Start Audiobook Synthesis")
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start)
        sizer.Add(self.start_button, 0, wx.ALL, 5)

        # Add Stop button
        # self.stop_button = wx.Button(panel, label="⏹️ Stop Synthesis")
        # self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop)
        # sizer.Add(self.stop_button, 0, wx.ALL, 5)
        # self.stop_button.Hide()

        # Add Progress Bar label:
        self.progress_bar_label = wx.StaticText(panel, label="Synthesis Progress:")
        sizer.Add(self.progress_bar_label, 0, wx.ALL, 5)
        self.progress_bar = CustomGauge(panel, range_val=100)
        self.progress_bar.SetMinSize((-1, 30))
        sizer.Add(self.progress_bar, 0, wx.ALL | wx.EXPAND, 5)
        self.progress_bar_label.Hide()
        self.progress_bar.Hide()

        # Add ETA Label
        self.eta_label = wx.StaticText(panel, label="Estimated Time Remaining: ")
        self.eta_label.Hide()
        sizer.Add(self.eta_label, 0, wx.ALL, 5)

        # Add Custom Rate input
        custom_rate_label = wx.StaticText(panel, label="Custom Rate (chars/sec, experimental):")
        sizer.Add(custom_rate_label, 0, wx.ALL, 5)

        saved_custom_rate = self.user_settings.get('custom_rate')
        initial_custom_rate_value = ""
        if saved_custom_rate is not None:
            try:
                self.custom_rate = int(saved_custom_rate)
                initial_custom_rate_value = str(self.custom_rate)
            except ValueError:
                self.custom_rate = None # Or a default int like 750
                print(f"Warning: Could not parse saved custom_rate '{saved_custom_rate}'")
        else:
            self.custom_rate = None # Default if not set

        self.custom_rate_text_ctrl = wx.TextCtrl(panel, value=initial_custom_rate_value)
        self.custom_rate_text_ctrl.Bind(wx.EVT_TEXT, self.on_set_custom_rate)
        sizer.Add(self.custom_rate_text_ctrl, 0, wx.ALL | wx.EXPAND, 5)

    def open_output_folder_dialog(self, event):
        result, output_folder = show_themed_dir_dialog(self, "Choose a directory:")
        if result == wx.ID_OK and output_folder:
            print(f"Selected output folder: {output_folder}")
            self.output_folder_text_ctrl.SetValue(output_folder)

    def on_set_custom_rate(self, event):
        rate_str = event.GetString()
        if not rate_str: # Empty input
            self.custom_rate = None
            db.save_user_setting('custom_rate', None) # Use db prefix
            print("Custom rate cleared and saved.")
            return

        try:
            rate = int(rate_str)
            if rate > 0:
                self.custom_rate = rate
                db.save_user_setting('custom_rate', self.custom_rate) # Use db prefix
                print(f"Custom rate set to {self.custom_rate} and saved.")
            # else: # Negative or zero, could show an error or ignore
            #    print(f"Invalid custom rate (must be positive): {rate_str}")
            #    if self.custom_rate is not None: # Reset to last valid or None
            #       self.custom_rate_text_ctrl.SetValue(str(self.custom_rate) if self.custom_rate else "")
            #    else:
            #       self.custom_rate_text_ctrl.SetValue("")

        except ValueError:
            # Non-integer input, ignore for now or show error
            print(f"Invalid custom rate input (must be an integer): {rate_str}")
            # Optionally reset text ctrl to last valid self.custom_rate or empty
            # if self.custom_rate is not None:
            #    self.custom_rate_text_ctrl.SetValue(str(self.custom_rate))
            # else:
            #    self.custom_rate_text_ctrl.SetValue("")


    def _load_book_data_into_ui(self, book_title, book_author, document_chapters, source_path, book_object=None, cover_info=None):
        # Cleanup previous dynamic UI parts if they exist
        # The UI is now static. We just enable the right panel and load data.
        if hasattr(self, 'splitter_right'):
            self.splitter_right.Enable()

        # Set instance variables
        self.selected_book_title = book_title
        self.selected_book_author = book_author
        self.document_chapters = document_chapters
        self.selected_file_path = source_path
        self.selected_book = book_object

        # Determine "good" chapters. For standard EPUBs, we find them.
        # For other sources (Calibre/Queue), we assume they are pre-selected.
        if self.selected_book:
            from audiblez.core import find_good_chapters
            self.good_chapters_list = find_good_chapters(self.document_chapters)
        else:
            self.good_chapters_list = [ch for ch in self.document_chapters if getattr(ch, 'is_selected', False)]

        # Process all chapters: set short_name and ensure is_selected is set.
        for chapter in self.document_chapters:
            # Set a user-friendly short_name if it doesn't exist
            if not getattr(chapter, 'short_name', ''):
                if hasattr(chapter, 'get_name'):
                    chapter.short_name = chapter.get_name().replace('.xhtml', '').replace('xhtml/', '').replace('.html', '').replace('Text/', '')
                elif hasattr(chapter, 'title'):
                    chapter.short_name = chapter.title
                else:
                    chapter.short_name = "Unknown Chapter"

            # Set selection status. For EPUBs, this is the first time.
            # For others, it's a confirmation based on good_chapters_list.
            chapter.is_selected = chapter in self.good_chapters_list

        # Determine selected_chapter based on good_chapters_list or document_chapters
        if self.good_chapters_list:
            self.selected_chapter = self.good_chapters_list[0]
        elif self.document_chapters:
            self.selected_chapter = self.document_chapters[0]
        else:
            self.selected_chapter = None

        # Create/ensure notebook and tab structure exists
        self.create_notebook_and_tabs()

        # 1. Populate the "Chapters" tab
        for child in self.chapters_tab_page.GetChildren():
            child.Destroy()
        self.chapters_panel = self.create_chapters_table_panel(self.document_chapters)
        chapters_page_sizer = wx.BoxSizer(wx.VERTICAL)
        chapters_page_sizer.Add(self.chapters_panel, 1, wx.EXPAND | wx.ALL)
        self.chapters_tab_page.SetSizer(chapters_page_sizer)
        self.chapters_tab_page.Layout()

        # 2. Populate the static panels with book data
        self.title_text.SetLabel(book_title)
        self.author_text.SetLabel(book_author)
        total_len = sum([len(c.extracted_text) for c in self.document_chapters])
        self.length_text.SetLabel(f'{total_len:,} characters')
        if self.selected_chapter:
            self.chapter_label.SetLabel(f'Edit / Preview content for section "{self.selected_chapter.short_name}":')
            self.text_area.SetValue(self.selected_chapter.extracted_text)
        else:
            self.chapter_label.SetLabel("No chapter selected.")
            self.text_area.SetValue("")

        # Update Cover
        if hasattr(self, 'cover_bitmap'):
            cover_image_to_load = None
            if cover_info:
                if cover_info['type'] == 'epub_cover' and cover_info['content']:
                    try:
                        pil_image = Image.open(io.BytesIO(cover_info['content']))
                        cover_image_to_load = pil_image
                    except Exception as e:
                        print(f"Error loading cover from epub content: {e}")
                elif cover_info['type'] == 'path' and cover_info['content'] and Path(cover_info['content']).exists():
                    try:
                        pil_image = Image.open(cover_info['content'])
                        cover_image_to_load = pil_image
                    except Exception as e:
                        print(f"Error loading cover from path {cover_info['content']}: {e}")

            if cover_image_to_load:
                try:
                    wx_img = wx.Image(cover_image_to_load.size[0], cover_image_to_load.size[1])
                    if cover_image_to_load.mode == 'RGBA':
                        pil_image_rgb = cover_image_to_load.convert('RGB')
                        wx_img.SetData(pil_image_rgb.tobytes())
                    else:
                        wx_img.SetData(cover_image_to_load.convert("RGB").tobytes())

                    cover_h = 200
                    cover_w = int(cover_h * cover_image_to_load.size[0] / cover_image_to_load.size[1])
                    if cover_w > 0 and cover_h > 0:
                        wx_img = wx_img.Scale(cover_w, cover_h, wx.IMAGE_QUALITY_HIGH)

                    self.cover_bitmap.SetBitmap(wx_img.ConvertToBitmap())
                    self.cover_bitmap.SetMaxSize((200, cover_h))
                except Exception as e_cover:
                    print(f"Error processing or displaying cover image: {e_cover}")
                    self.cover_bitmap.SetBitmap(wx.NullBitmap)
            else:
                self.cover_bitmap.SetBitmap(wx.NullBitmap)

        self.refresh_staging_tab()
        self.refresh_queue_tab()

        self.splitter.Layout()
        self.Layout()


    def refresh_queue_tab(self):
        # Clear existing content from the queue_tab_panel's sizer
        # print(f"DEBUG: refresh_queue_tab called. self.queue_items: {self.queue_items}")
        def _load_epub_file(self, file_path):
            """Helper function to load and process an EPUB file."""
            print(f"Opening file: {file_path}")
    
            from ebooklib import epub
            from audiblez.core import find_document_chapters_and_extract_texts, find_cover
            from pathlib import Path
    
            try:
                book = epub.read_epub(file_path)
                meta_title = book.get_metadata('DC', 'title')
                title = meta_title[0][0] if meta_title else Path(file_path).stem
                meta_creator = book.get_metadata('DC', 'creator')
                author = meta_creator[0][0] if meta_creator else 'Unknown Author'
    
                document_chapters = find_document_chapters_and_extract_texts(book)
    
                cover = find_cover(book)
                cover_info = {'type': 'epub_cover', 'content': cover.content} if cover else None
    
                # The UI update should happen on the main thread
                wx.CallAfter(self._load_book_data_into_ui,
                    book_title=title,
                    book_author=author,
                    document_chapters=document_chapters,
                    source_path=file_path,
                    book_object=book,
                    cover_info=cover_info
                )
            except Exception as e:
                print(f"Error opening EPUB file {file_path}: {e}")
                show_themed_message(self, f"Failed to open or parse the EPUB file:\n\n{e}", "EPUB Error", wx.OK | wx.ICON_ERROR)

        if hasattr(self, 'queue_tab_sizer') and self.queue_tab_sizer:
            # Clear the sizer and delete all windows it managed.
            # This is the most common and robust way to reset a sizer's content.
            self.queue_tab_sizer.Clear(delete_windows=True)
            # print("DEBUG: Called self.queue_tab_sizer.Clear(delete_windows=True)")
            # Any windows previously in the sizer (like individual queue item boxes,
            # the 'empty queue' text, or the run_queue_button if it was part of it)
            # are now destroyed. They will be recreated as needed below.
            # If self.run_queue_button was a child and was part of this sizer,
            # it is now destroyed, and self.run_queue_button would be a stale reference.
            # The existing logic for creating/showing the button later in this method
            # (e.g., `if not self.run_queue_button:`) should ideally handle
            # the recreation if self.run_queue_button becomes None or if operations on a stale
            # reference lead to expected errors that are gracefully handled.
            # For now, we rely on Clear() to do its job and the subsequent button logic
            # to correctly reconstruct or re-add the button.

        if not self.queue_items:
            no_items_label = wx.StaticText(self.queue_tab_panel, label="The synthesis queue is empty.")
            self.queue_tab_sizer.Add(no_items_label, 0, wx.ALL | wx.ALIGN_CENTER, 15)
            # print("DEBUG: Queue is empty, adding placeholder label.")
        else:
            for item_idx, item_data in enumerate(self.queue_items):
                # print(f"DEBUG: Processing queue item {item_idx}: {item_data.get('book_title')}")
                # Main container for each queue item
                item_box_label = f"#{item_idx + 1}: {item_data['book_title']}"
                # Add status to the label if present
                current_status = item_data.get('status', 'Pending')
                if self.queue_processing_active and item_idx == self.current_queue_item_index:
                    current_status = item_data.get('status', "⏳ In Progress") # Default to In Progress if it's the current one

                item_display_label = f"{item_box_label} - Status: {current_status}"
                item_container = wx.Panel(self.queue_tab_panel, style=wx.BORDER_THEME)
                item_sizer = wx.BoxSizer(wx.VERTICAL)
                item_container.SetSizer(item_sizer)

                label = wx.StaticText(item_container, label=item_display_label)
                font = label.GetFont()
                font.SetWeight(wx.FONTWEIGHT_BOLD)
                label.SetFont(font)
                item_sizer.Add(label, 0, wx.ALL, 5)

                # Chapters information
                chapters_str = "All Chapters"  # Default if specific chapters aren't listed (e.g. whole book)
                if 'chapters' in item_data and isinstance(item_data['chapters'], list):
                    if len(item_data['chapters']) > 3:
                        chapters_str = f"Selected chapters ({len(item_data['chapters'])})"
                    else:
                        chapters_str = ", ".join([ch['title'] for ch in item_data['chapters']])
                    if not chapters_str: chapters_str = "No specific chapters selected"
                # Remove legacy/old chapter handling. Only support new chapter structure.
                # (No action needed, just skip legacy logic.)


                chapters_label = wx.StaticText(item_container, label=f"Chapters: {chapters_str}")
                item_sizer.Add(chapters_label, 0, wx.ALL | wx.EXPAND, 5)

                # Synthesis settings
                settings = item_data.get('synthesis_settings', {})
                engine_label = wx.StaticText(item_container, label=f"Engine: {settings.get('engine', 'N/A')}")
                output_label = wx.StaticText(item_container, label=f"Output: {settings.get('output_folder', 'N/A')}")
                output_label.Wrap(self.window_width // 3)  # Wrap text if too long

                item_sizer.Add(engine_label, 0, wx.ALL | wx.EXPAND, 2)
                item_sizer.Add(output_label, 0, wx.ALL | wx.EXPAND, 2)

                # Store a reference to the container panel in the item_data if needed for updates
                item_data['_ui_box'] = item_container

                # Add Remove button for each item
                remove_button = wx.Button(item_container, label="❌ Remove")
                # Pass queue_item_id (item_data['id']) to the handler
                # Ensure item_data['id'] exists and is the correct DB ID for the queue item
                if 'id' in item_data:
                    remove_button.Bind(wx.EVT_BUTTON, lambda evt, qid=item_data['id']: self.on_remove_queue_item(evt, qid))
                else:
                    remove_button.Disable() # Should not happen if items are from DB
                    print(f"Warning: Queue item '{item_data.get('book_title')}' is missing an 'id'. Remove button disabled.")
                item_sizer.Add(remove_button, 0, wx.ALL | wx.ALIGN_CENTER, 5)

                self.queue_tab_sizer.Add(item_container, 0, wx.ALL | wx.EXPAND, 10)
                # print(f"DEBUG: Added item_sizer for {item_data.get('book_title')} to queue_tab_sizer.")

        # Sizer for action buttons (Run, Schedule) and scheduled time text
        action_controls_sizer = wx.BoxSizer(wx.HORIZONTAL)

        if self.queue_items:
            # Create buttons and text FRESHLY each time, as Clear(delete_windows=True) destroyed old ones.
            self.run_queue_button = wx.Button(self.queue_tab_panel, label="🚀 Run Queue")
            self.run_queue_button.Bind(wx.EVT_BUTTON, self.on_run_queue)
            action_controls_sizer.Add(self.run_queue_button, 0, wx.ALL | wx.ALIGN_CENTER, 5)
            self.run_queue_button.Enable(not self.queue_processing_active)

            self.schedule_queue_button = wx.Button(self.queue_tab_panel, label="📅 Schedule Queue")
            self.schedule_queue_button.Bind(wx.EVT_BUTTON, self.on_schedule_queue)
            action_controls_sizer.Add(self.schedule_queue_button, 0, wx.ALL | wx.ALIGN_CENTER, 5)
            self.schedule_queue_button.Enable(not self.queue_processing_active)

            self.scheduled_time_text = wx.StaticText(self.queue_tab_panel, label="")
            action_controls_sizer.Add(self.scheduled_time_text, 0, wx.ALL | wx.ALIGN_CENTER | wx.LEFT, 10)
            self.update_scheduled_time_display() # This will set the label for scheduled_time_text

            if action_controls_sizer.GetItemCount() > 0:
                self.queue_tab_sizer.Add(action_controls_sizer, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        else:
            # Ensure instance variables are None if controls are not created / relevant
            self.run_queue_button = None
            self.schedule_queue_button = None
            self.scheduled_time_text = None
            # Any existing action_controls_sizer (if it was managed at instance level)
            # would have been cleared by self.queue_tab_sizer.Clear(delete_windows=True)
            # if it was added to queue_tab_sizer. Here, action_controls_sizer is local.

        # print("DEBUG: Calling self.queue_tab_panel.SetupScrolling() and .Layout()")
        self.queue_tab_panel.SetupScrolling()
        self.queue_tab_panel.Layout()
        if hasattr(self, 'notebook') and self.notebook:
            # print("DEBUG: Calling self.notebook.Layout() and self.splitter_left.Layout()")
            self.notebook.Layout()
        if hasattr(self, 'splitter_left') and self.splitter_left:
            self.splitter_left.Layout()
        # self.Layout() # Optionally, layout the whole frame if needed
        self.apply_theme(self.theme_name)

    def update_scheduled_time_display(self):
        if not hasattr(self, 'scheduled_time_text') or not self.scheduled_time_text:
            return
        scheduled_ts = db.load_schedule_time()
        if scheduled_ts:
            try:
                if scheduled_ts > 0: # Valid timestamp
                    scheduled_dt = datetime.fromtimestamp(scheduled_ts)
                    self.scheduled_time_text.SetLabel(f"Scheduled for: {scheduled_dt.strftime('%Y-%m-%d %H:%M')}")
                else: # Invalid or cleared timestamp
                    self.scheduled_time_text.SetLabel("Not scheduled")
            except (TypeError, ValueError, OSError) as e: # Catch potential errors from invalid timestamp
                self.scheduled_time_text.SetLabel("Scheduled: (Error)")
                print(f"Error formatting scheduled time (ts: {scheduled_ts}): {e}")
        else:
            self.scheduled_time_text.SetLabel("Not scheduled")

        if self.scheduled_time_text.GetContainingSizer():
            self.scheduled_time_text.GetContainingSizer().Layout()
        # self.queue_tab_panel.Layout() # Avoid if too broad, parent sizer should handle

    def on_schedule_queue(self, event):
        dialog = ScheduleDialog(self)
        if dialog.ShowModal() == wx.ID_OK:
            selected_datetime = dialog.get_selected_datetime()
            if selected_datetime: # A specific datetime was chosen
                if selected_datetime < datetime.now():
                    show_themed_message(self, "Scheduled time must be in the future.", "Invalid Time", wx.OK | wx.ICON_ERROR)
                    dialog.Destroy()
                    return

                timestamp = int(selected_datetime.timestamp())
                db.save_schedule_time(timestamp)
                show_themed_message(self, f"Queue scheduled to run at: {selected_datetime.strftime('%Y-%m-%d %H:%M')}",
                              "Queue Scheduled", wx.OK | wx.ICON_INFORMATION)
                # self.start_schedule_check_timer() # Call to start timer will be added later
            else: # User explicitly cleared the schedule via the dialog
                db.save_schedule_time(None) # Pass None to clear
                show_themed_message(self, "Queue schedule has been cleared.", "Schedule Cleared", wx.OK | wx.ICON_INFORMATION)
                # if hasattr(self, 'schedule_check_timer') and self.schedule_check_timer.IsRunning():
                #     self.schedule_check_timer.Stop()
                #     print("Schedule check timer stopped.")
            self.update_scheduled_time_display() # Update display after any change
        dialog.Destroy()

    def start_schedule_check_timer(self, interval_ms=30000): # Check every 30 seconds
        """Starts or restarts the schedule check timer."""
        if self.schedule_check_timer.IsRunning():
            self.schedule_check_timer.Stop()

        # Only start if there's actually a schedule to check or if we want it always running
        # For now, let's make it always start and on_check_schedule_timer can decide to do nothing
        print(f"Schedule check timer started/restarted (interval: {interval_ms}ms).")
        self.schedule_check_timer.Start(interval_ms)


    def on_check_schedule_timer(self, event):
        # print("Checking schedule...") # For debugging
        if self.queue_processing_active:
            # print("Queue is busy, skipping scheduled check.")
            return

        scheduled_ts = db.load_schedule_time()
        if not scheduled_ts or scheduled_ts <= 0: # No schedule or invalid
            # print("No active schedule found.")
            # self.schedule_check_timer.Stop() # Optional: stop if no schedule
            return

        current_ts = int(datetime.now().timestamp())
        if current_ts >= scheduled_ts:
            print(f"Scheduled time {datetime.fromtimestamp(scheduled_ts)} reached. Starting queue.")
            db.save_schedule_time(None) # Clear the schedule from DB
            self.update_scheduled_time_display() # Update UI

            if not self.queue_items:
                show_themed_message(self, "Scheduled time reached, but the queue is empty.",
                              "Queue Empty", wx.OK | wx.ICON_INFORMATION)
                if self.schedule_check_timer.IsRunning(): self.schedule_check_timer.Stop() # Stop timer
                return

            # Check if another synthesis (manual or other) started in the meantime
            if self.synthesis_in_progress or self.queue_processing_active:
                 print("Synthesis/Queue processing started by other means before scheduled time could trigger. Schedule ignored.")
                 return

            self.on_run_queue(event=None) # Trigger queue processing
            # Timer will continue running, or could be stopped if preferred after a run
            # For now, let it run; it won't do anything until a new schedule is set.
        # else:
            # print(f"Scheduled time {datetime.fromtimestamp(scheduled_ts)} not yet reached.")

    def _get_calibre_details_for_queued_item(self, item_data: dict) -> tuple[dict | None, str | None]:
        """
        Helper to retrieve Calibre metadata and cover path for a QUEUED item.
        This is tricky because the queue item might have been added from a Calibre-imported book
        that is no longer the "current" book in the UI.
        We need to rely on information stored WITH the queue item.
        """
        # How 'book_data' (containing metadata and cover_path) is stored for queued Calibre items:
        # - When "Queue Selected Book Portions" is used for a Calibre-imported book currently in UI:
        #   The `self.book_data` (set by on_open_with_calibre) should ideally be copied into the queue_entry.
        # - When "Queue Selected Chapters" from Staging tab is used for a Calibre-imported book:
        #   The `db.add_staged_book` would need to store the metadata and cover_image_path from the original import.
        #   Then `db.add_item_to_queue` (when called from `on_queue_selected_staged_chapters`)
        #   would need to fetch this from the staged_book record and include it in the queue item's data.

        # Current implementation:
        # `on_queue_selected_book_portions` (from Chapters tab): Does NOT explicitly copy self.book_data.
        #   It relies on `source_path` being the original Calibre input.
        # `on_queue_selected_staged_chapters` (from Staging tab): Also relies on `source_path` if it were stored,
        #   or needs the `staged_book_id` to trace back to original metadata/cover.

        # For now, let's assume if `item_data['source_path']` was the original Calibre input
        # and `item_data` also has `calibre_metadata` and `calibre_cover_path` directly stored
        # (this needs to be added when item is queued).

        # Check if the item itself has these details (IDEAL for queued items)
        if isinstance(item_data.get('synthesis_settings'), dict) and \
           item_data['synthesis_settings'].get('calibre_metadata') is not None:
            print(f"Found Calibre details directly in queued item: {item_data['book_title']}")
            return item_data['synthesis_settings']['calibre_metadata'], item_data['synthesis_settings'].get('calibre_cover_image_path')

        # Fallback: If the item's source_path matches the currently loaded Calibre book in UI.
        # This is less reliable for queues but might work for items queued and run immediately.
        if hasattr(self, 'book_data') and self.book_data and \
           'metadata' in self.book_data and \
           item_data.get('source_path') == self.selected_file_path: # selected_file_path is the original input
            print(f"Using current UI's Calibre data for queued item: {item_data['book_title']}")
            return self.book_data['metadata'], self.book_data.get('cover_image_path')

        # If the queue item was from a staged book, and that staged book stored its original metadata/cover.
        # This requires `db.get_staged_book_details(item_data['staged_book_id'])` to return them.
        # This is not yet implemented in the DB schema for staged books.

        return None, None


    def on_remove_queue_item(self, event, queue_item_id):
        """Handles removal of a specific item from the queue."""
        if self.queue_processing_active:
            # Find the item being processed
            if self.current_queue_item_index >= 0 and self.current_queue_item_index < len(self.queue_items):
                currently_processing_item_id = self.queue_items[self.current_queue_item_index].get('id')
                if currently_processing_item_id == queue_item_id:
                    show_themed_message(self, "Cannot remove an item that is currently being processed.",
                                  "Action Not Allowed", wx.OK | wx.ICON_WARNING)
                    return

        # Optional: Confirmation dialog
        # confirm = show_themed_message(self, f"Are you sure you want to remove this item from the queue?",
        #                         "Confirm Removal", wx.YES_NO | wx.ICON_QUESTION)
        # if confirm == wx.NO:
        #     return

        db.remove_queue_item(queue_item_id)
        self.queue_items = db.get_queued_items() # Reload queue from DB
        self.refresh_queue_tab() # Refresh the UI display

        # If the removed item was before the currently processing one, adjust index
        # This is a bit tricky if items are reordered or if current_queue_item_index
        # refers to the old list. Simplest is to let process_next_queue_item handle it,
        # or re-evaluate current_queue_item_index based on the new list if processing.
        # For now, if queue is active, it might try to process an item that shifted index.
        # However, remove_queue_item is typically for non-active queues or items not yet processed.

        show_themed_message(self, "Item removed from queue.", "Queue Updated", wx.OK | wx.ICON_INFORMATION)


    def on_run_queue(self, event):
        if not self.queue_items:
            show_themed_message(self, "Queue is empty. Add items to the queue first.", "Queue Empty", wx.OK | wx.ICON_INFORMATION)
            return

        if self.queue_processing_active:
            show_themed_message(self, "Queue processing is already active.", "Queue Running", wx.OK | wx.ICON_INFORMATION)
            return

        # Clear any existing schedule if queue is run manually
        if db.load_schedule_time():
            db.save_schedule_time(None) # Clear schedule from DB
            self.update_scheduled_time_display() # Update UI
            print("Manual queue run initiated, existing schedule cleared.")

        self.queue_processing_active = True
        self.current_queue_item_index = 0 # Start with the first item
        # self.queue_to_process = list(self.queue_items) # Process a copy

        if self.run_queue_button:
            self.run_queue_button.Disable()
        if self.schedule_queue_button: # Also disable schedule button
            self.schedule_queue_button.Disable()


        if hasattr(self, 'start_button'):
            self.start_button.Disable() # Disable single start button as well
        if hasattr(self, 'params_panel'):
            self.params_panel.Disable() # Disable params panel

        self.process_next_queue_item()

    def process_next_queue_item(self):
        if not self.queue_processing_active: # Stopped externally
            self._finalize_queue_processing()
            return

        if self.current_queue_item_index >= len(self.queue_items):
            show_themed_message(self, "All items in the queue have been processed.", "Queue Finished", wx.OK | wx.ICON_INFORMATION)
            self._finalize_queue_processing()
            return

        item_to_process = self.queue_items[self.current_queue_item_index]
        # Update DB status and local status
        db.update_queue_item_status(item_to_process['id'], 'in_progress')
        item_to_process['status'] = "⏳ In Progress"
        self.refresh_queue_tab()

        book_title = item_to_process['book_title']
        # synthesis_settings is already a dict due to db.get_queued_items()
        synthesis_settings = item_to_process['synthesis_settings']
        chapters_to_synthesize = []

        # Prepare chapters for CoreThread: ensure they have 'extracted_text' and 'chapter_index'
        # item_to_process['chapters'] comes from db.get_queued_items which gets from queued_chapters table
        for idx, chap_db_info in enumerate(item_to_process.get('chapters', [])):
            chapter_obj = type('ChapterForCore', (), {})()
            title = chap_db_info.get('title', 'Unknown Chapter')
            chapter_obj.title = title
            chapter_obj.get_name = lambda t=title: t  # Mimic ebooklib chapter for core.py compatibility
            chapter_obj.short_name = title # For consistency if core uses short_name
            chapter_obj.chapter_index = idx # Index within this synthesis job for UI event

            text_content = chap_db_info.get('text_content')
            # If text_content is None or empty, and there's a staged_chapter_id, try fetching it
            if not text_content and chap_db_info.get('staged_chapter_id'):
                print(f"Fetching text for staged chapter ID {chap_db_info['staged_chapter_id']} ('{chapter_obj.title}')...")
                text_content = db.get_chapter_text_content(chap_db_info['staged_chapter_id'])

            if text_content is None: # If still None after trying to fetch
                print(f"Error: Could not find/fetch text for chapter ID {chap_db_info.get('id')} ('{chapter_obj.title}') in book '{book_title}'. Skipping chapter.")
                # Optionally mark chapter or item as error here. For now, just skip this chapter.
                continue
            chapter_obj.extracted_text = text_content
            chapter_obj.is_selected = True # All chapters here are for processing
            # Ensure 'title' is set for chapter filename generation in core.main if 'short_name' isn't what's needed
            if not hasattr(chapter_obj, 'title') or not chapter_obj.title:
                 chapter_obj.title = chap_db_info.get('title', f'Chapter_{idx+1}')
            chapters_to_synthesize.append(chapter_obj)

        if not chapters_to_synthesize:
            print(f"Skipping '{book_title}': No valid chapters to synthesize after attempting to load text.")
            db.update_queue_item_status(item_to_process['id'], 'error') # Mark as error in DB
            item_to_process['status'] = "⚠️ Error (No Chapters Text)" # Update local status
            self.refresh_queue_tab() # Refresh UI
            self.current_queue_item_index += 1 # Move to next item
            wx.CallAfter(self.process_next_queue_item) # Try next item
            return

        # Prepare parameters for CoreThread
        # `file_path` is the original input file path (e.g., mybook.epub, mybook.mobi)
        # It's used by core.main for output filename generation.
        file_path = item_to_process.get('source_path')

        # For staged items, 'source_path' might be in the main item_to_process (if queued from Chapters tab initially)
        # or associated with the staged_book_id in the database.
        # The `db.add_item_to_queue` for staged items currently sets `source_path` to None.
        # We need a reliable way to get the original input filename for `core.main`.
        if not file_path and item_to_process.get('staged_book_id'):
            # Try to get source_path from the staged book's record in DB
            staged_book_details = db.get_staged_book_details(item_to_process['staged_book_id'])
            if staged_book_details and staged_book_details.get('source_path'):
                file_path = staged_book_details['source_path']
                print(f"Using source_path '{file_path}' from staged book record for '{book_title}'.")
            else:
                # Fallback: use book title to create a dummy filename if source_path is truly unavailable.
                # This ensures core.main has *a* file_path for naming.
                safe_book_title = "".join(c if c.isalnum() else "_" for c in book_title)
                file_path = f"{safe_book_title}.calibre_import" # Dummy extension
                print(f"Warning: Original source_path not found for staged item '{book_title}'. Using dummy file_path: '{file_path}' for output naming.")


        if not file_path and 'book_id' in item_to_process: # Staged item, old check, should be covered by above
            # Attempt to use current book's path as a fallback for core.main's epub.read_epub.
            # This is primarily so core.main doesn't crash trying to read a non-existent/placeholder path.
            # The actual chapter content for TTS comes from chapters_to_synthesize.
            # Metadata (title, author, cover) in the output file might be from the wrong book if it's not a match.
            # This will be properly fixed when DB stores source_path or metadata for staged items.
            if hasattr(self, 'selected_file_path') and self.selected_file_path and Path(self.selected_file_path).exists():
                file_path = self.selected_file_path
                print(f"Warning: Using currently loaded EPUB '{file_path}' as a source for metadata for staged item '{book_title}'. Output metadata may be incorrect if this is not the original EPUB for the staged item.")
            else:
                # If no current EPUB is loaded, core.main will likely fail.
                # This is an unrecoverable situation for core.main as it's currently written.
                print(f"Error: Cannot process staged item '{book_title}'. Original source_path is missing and no fallback EPUB is currently loaded. Skipping.")
                item_to_process['status'] = "⚠️ Error (Missing EPUB Path)"
                self.current_queue_item_index +=1
                # self.synthesis_in_progress = False # Reset as this item won't run CoreThread
                wx.CallAfter(self.process_next_queue_item)
                return
        elif not file_path: # Should not happen if logic is correct (source_path from Chapters tab)
             print(f"Error: file_path is missing for item '{book_title}' and it's not a staged item. Skipping.")
             item_to_process['status'] = "⚠️ Error (Missing Path)"
             self.current_queue_item_index +=1
             wx.CallAfter(self.process_next_queue_item)
             return

        # NEW: Load this book's data into the UI
        # This ensures the chapter table is correct for status updates.
        # We need to find the author from the staged book details if available
        author = "From Queue"
        if item_to_process.get('staged_book_id'):
            staged_book_details = db.get_staged_book_details(item_to_process['staged_book_id'])
            if staged_book_details and staged_book_details.get('author'):
                author = staged_book_details.get('author')

        wx.CallAfter(self._load_book_data_into_ui,
            book_title=book_title,
            book_author=author,
            document_chapters=chapters_to_synthesize,
            source_path=file_path,
            book_object=None,
            cover_info=None # Cover not stored in queue, can be added later
        )
        # A small delay to allow the UI to update before processing starts
        import time
        time.sleep(0.2)


        def fail_item(reason):
            """Helper to fail the current queue item and move to the next."""
            print(f"Skipping '{book_title}': {reason}.")
            db.update_queue_item_status(item_to_process['id'], 'error')
            item_to_process['status'] = f"⚠️ Error ({reason})"
            self.refresh_queue_tab()
            self.current_queue_item_index += 1
            wx.CallAfter(self.process_next_queue_item)


        output_folder = synthesis_settings.get('output_folder')
        if not output_folder:
            fail_item("Missing 'output_folder' setting")
            return

        engine = synthesis_settings.get('engine', 'cpu') # Default to CPU if not specified

        # Set device for this specific core.main call
        torch.set_default_device(engine)
        print(f"Setting engine to: {engine} for book: {book_title}")


        if not chapters_to_synthesize:
             print(f"No chapters to synthesize for '{book_title}'. Skipping.")
             item_to_process['status'] = "⚠️ Skipped (No Chapters)"
             self.current_queue_item_index +=1
             wx.CallAfter(self.process_next_queue_item)
             return

        print(f"Starting synthesis from queue for: {book_title} with {len(chapters_to_synthesize)} chapters.")
        self.synthesis_in_progress = True # General flag for core processing

        # Ensure UI elements like progress bar are visible for the current item
        self.progress_bar_label.SetLabel(f"Progress for: {book_title}")
        self.progress_bar_label.Show()
        self.progress_bar.SetValue(0)
        self.progress_bar.Show()
        self.eta_label.Show()
        self.synth_panel.Layout()


        # Note: CoreThread's post_event uses chapter.chapter_index.
        # Make sure chapters_to_synthesize have this attribute.

        core_params = {
            'file_path': file_path,
            'pick_manually': False,
            'output_folder': output_folder,
            'selected_chapters': chapters_to_synthesize,
            'calibre_metadata': None,
            'calibre_cover_image_path': None,
            'm4b_assembly_method': synthesis_settings.get('m4b_assembly_method', self.m4b_assembly_method)
        }

        # Try to get Calibre-specific details for this queued item
        # This helper function encapsulates the logic to find them if they exist for this item.
        # Modification needed: `_get_calibre_details_for_queued_item` needs to be robust.
        # It should check if the item was originally a Calibre import and if its metadata/cover path
        # were stored with the queue item or can be retrieved via staged_book_id.

        # For the purpose of this step, we assume `item_to_process` MIGHT have these if stored during queueing.
        # This part needs to be solidified when queueing Calibre books:
        # Ensure 'calibre_metadata_override' and 'calibre_cover_path_override' are stored in `item_to_process['synthesis_settings']` if applicable.
        queued_calibre_meta = item_to_process.get('synthesis_settings', {}).get('calibre_metadata_override')
        queued_calibre_cover = item_to_process.get('synthesis_settings', {}).get('calibre_cover_path_override')

        if queued_calibre_meta:
            core_params['calibre_metadata'] = queued_calibre_meta
            core_params['calibre_cover_image_path'] = queued_calibre_cover # May be None
            print(f"Using Calibre metadata/cover override from queued item for: {book_title}")

        self.core_thread = CoreThread(params=core_params)
        self.core_thread.start()
        # The actual removal from self.queue_items and moving to next happens in on_core_finished

    def _finalize_queue_processing(self):
        self.queue_processing_active = False
        self.synthesis_in_progress = False # Reset general flag
        self.current_queue_item_index = -1
        # self.queue_to_process = [] # Clear the processing copy

        # Ensure schedule display is up-to-date (e.g. if it was cleared by starting queue)
        self.update_scheduled_time_display()

        # Remove items from DB that were successfully processed or errored out
        items_to_remove_from_db = [
            item['id'] for item in self.queue_items
            if 'id' in item and (item.get('status', '').startswith("✅") or \
                                 item.get('status', '').startswith("⚠️"))
        ]
        for item_id_to_remove in items_to_remove_from_db:
            db.remove_queue_item(item_id_to_remove)

        self.queue_items = db.get_queued_items() # Reload to get the current truth from DB
        self.refresh_queue_tab()

        # Re-enable global controls if no more items or queue stopped
        if hasattr(self, 'start_button'):
            self.start_button.Enable()
        if hasattr(self, 'params_panel'):
            self.params_panel.Enable()
        if hasattr(self, 'table'): self.table.Enable(True)

        if self.run_queue_button:
            if self.queue_items: # If some items remain (e.g. user added more)
                self.run_queue_button.Enable()
            else: # Queue is now empty
                self.run_queue_button.Disable() # Or remove, handled by refresh_queue_tab

        # Hide progress bar elements related to single/queue item processing
        self.progress_bar_label.Hide()
        self.progress_bar.Hide()
        self.eta_label.Hide()
        self.params_panel.Layout() # Was disabled
        self.synth_panel.Layout()


    def on_table_checked(self, event):
        self.document_chapters[event.GetIndex()].is_selected = True

    def on_table_unchecked(self, event):
        self.document_chapters[event.GetIndex()].is_selected = False

    def on_table_selected(self, event):
        chapter = self.document_chapters[event.GetIndex()]
        print('Selected', event.GetIndex(), chapter.short_name)
        self.selected_chapter = chapter
        self.text_area.SetValue(chapter.extracted_text)
        self.chapter_label.SetLabel(f'Edit / Preview content for section "{chapter.short_name}":')

    def create_chapters_table_panel(self, document_chapters_list):
        # Parent of this ScrolledPanel should be self.chapters_tab_page (the wx.Panel for "Chapters" tab)
        panel = ScrolledPanel(self.chapters_tab_page, -1, style=wx.TAB_TRAVERSAL | wx.SUNKEN_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)

        agwStyle = ULC_REPORT | ULC_SINGLE_SEL
        self.table = table = UltimateListCtrl(panel, agwStyle=agwStyle)
        table.InsertColumn(0, "Included", width=80)
        table.InsertColumn(1, "Chapter Name", width=150)
        table.InsertColumn(2, "Chapter Length", width=150)
        table.InsertColumn(3, "Status", width=100)

        table.Bind(wx.EVT_LIST_ITEM_CHECKED, self.on_table_checked)
        table.Bind(wx.EVT_LIST_ITEM_UNCHECKED, self.on_table_unchecked)
        table.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_table_selected)

        # self.good_chapters_list should be available here, set in open_epub
        for i, chapter in enumerate(document_chapters_list):  # Use the passed argument
            auto_selected = chapter.is_selected
            # ULC uses InsertStringItem with it_kind=1 to create a checkbox item
            index = table.InsertStringItem(i, "", it_kind=1)
            table.SetStringItem(index, 1, chapter.short_name)
            table.SetStringItem(index, 2, f"{len(chapter.extracted_text):,}")
            table.SetStringItem(index, 3, "") # Status column
            if auto_selected:
                item = table.GetItem(index)
                item.Check(True)
                table.SetItem(item)
            # Ensure chapter.is_selected is consistent if it wasn't set by open_epub's loop,
            # though it should be. This is more about table checking.
            # chapter.is_selected = auto_selected # This line is redundant if open_epub already set it.

        title_text = wx.StaticText(panel, label=f"Select chapters to include in the audiobook:")
        sizer.Add(title_text, 0, wx.ALL, 5)
        sizer.Add(table, 1, wx.ALL | wx.EXPAND, 5)

        stage_book_button = wx.Button(panel, label="📚 Stage Book for Batching")
        stage_book_button.Bind(wx.EVT_BUTTON, self.on_stage_book)
        sizer.Add(stage_book_button, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        queue_portions_button = wx.Button(panel, label="▶️ Queue Selected Book Portions")
        queue_portions_button.Bind(wx.EVT_BUTTON, self.on_queue_selected_book_portions)
        sizer.Add(queue_portions_button, 0, wx.ALL | wx.ALIGN_CENTER, 10)

        return panel

    def on_queue_selected_book_portions(self, event):
        # A book is considered loaded if it's a regular EPUB (self.selected_book)
        # or if it's from Calibre (self.book_data).
        book_is_loaded = (hasattr(self, 'selected_book') and self.selected_book) or \
                         (hasattr(self, 'book_data') and self.book_data)

        if not book_is_loaded:
            show_themed_message(self, "Please open an EPUB file first to select and queue book portions.",
                          "No Book Loaded", wx.OK | wx.ICON_INFORMATION)
            return

        selected_chapters_from_table = []
        for i in range(self.table.GetItemCount()):
            if self.table.IsItemChecked(i):
                # self.document_chapters[i] should correspond to the displayed item
                selected_chapters_from_table.append(self.document_chapters[i])

        if not selected_chapters_from_table:
            show_themed_message(self, "No chapters selected from the list. Please check the chapters you want to queue.",
                          "No Selection", wx.OK | wx.ICON_INFORMATION)
            return

        # Retrieve current global synthesis settings
        current_engine = 'cuda' if self.cuda_toggle.GetValue() else 'cpu'
        current_output_folder = self.output_folder_text_ctrl.GetValue()

        synthesis_settings = {
            'engine': current_engine,
            'output_folder': current_output_folder,
            # Initialize Calibre specific overrides to None
            'calibre_metadata_override': None,
            'calibre_cover_path_override': None,
            'm4b_assembly_method': self.m4b_assembly_method,
        }

        # If the current book in UI (self.selected_file_path) was from Calibre,
        # and has self.book_data, then pass this data for the queue item.
        if hasattr(self, 'book_data') and self.book_data and \
           'metadata' in self.book_data and self.selected_file_path == self.selected_file_path: # Ensure it's the current book
            # Check if 'metadata' indicates it's from Calibre (e.g., by a specific key or just assume if book_data exists fully)
            # For now, if self.book_data['metadata'] exists, we assume it's Calibre data to be passed.
            print(f"Adding Calibre-specific metadata and cover path to queue item for '{self.selected_book_title}' from Chapters tab.")
            synthesis_settings['calibre_metadata_override'] = self.book_data['metadata']
            synthesis_settings['calibre_cover_path_override'] = self.book_data.get('cover_image_path')


        # Create a new queue entry
        # Note: 'selected_chapter_details' stores the actual chapter objects from self.document_chapters
        # This is different from the Staging tab queue which stores DB IDs and titles.
        # The processor will need to handle this difference.
        queue_entry = {
            'staged_book_id': None, # Not from staging
            'book_title': self.selected_book_title,
            'source_path': self.selected_file_path,
            'synthesis_settings': synthesis_settings, # This is a dict
            'chapters': []
        }
        for i, chap_obj in enumerate(selected_chapters_from_table):
            queue_entry['chapters'].append({
                'staged_chapter_id': None,
                'title': chap_obj.short_name,
                'text_content': chap_obj.extracted_text, # Store text directly for non-staged items
                'order': i
            })

        # print(f"DEBUG: Before DB reload, self.queue_items: {self.queue_items}")
        new_item_id = db.add_item_to_queue(queue_entry)
        if new_item_id:
            self.queue_items = db.get_queued_items() # Reload queue
            # print(f"DEBUG: After DB reload, self.queue_items: {self.queue_items}")
            # print("DEBUG: Calling refresh_queue_tab()")
            self.refresh_queue_tab()
            self.notebook.SetSelection(self.notebook.GetPageCount() - 1)
            show_themed_message(self, f"Added selected portions from '{self.selected_book_title}' (with {len(selected_chapters_from_table)} chapter(s)) to the queue.",
                          "Added to Queue", wx.OK | wx.ICON_INFORMATION)
        else:
            show_themed_message(self, "Failed to add item to the database queue.", "Error", wx.OK | wx.ICON_ERROR)


    def on_stage_book(self, event):
        if not hasattr(self, 'selected_book') or not self.selected_book:
            show_themed_message(self, "Please open an EPUB file first.", "No Book Loaded", wx.OK | wx.ICON_INFORMATION)
            return

        book_title = self.selected_book_title
        book_author = self.selected_book_author
        source_path = self.selected_file_path
        output_folder = self.output_folder_text_ctrl.GetValue()

        chapters_to_stage = []
        for i, chapter_obj in enumerate(self.document_chapters):
            chapters_to_stage.append({
                'chapter_number': i, # or some other persistent chapter identifier if available
                'title': chapter_obj.short_name,
                'text_content': chapter_obj.extracted_text,
                'is_selected_for_synthesis': chapter_obj.is_selected
            })

        from audiblez.database import add_staged_book # Import moved here for clarity
        book_id = add_staged_book(book_title, book_author, source_path, output_folder, chapters_to_stage)

        if book_id is not None:
            show_themed_message(self, f"Book '{book_title}' and its chapters have been staged.", "Book Staged", wx.OK | wx.ICON_INFORMATION)
            self.refresh_staging_tab()
        elif source_path:
             show_themed_message(self, f"Book '{book_title}' (from {source_path}) might already be staged. Cannot add duplicate.", "Staging Failed", wx.OK | wx.ICON_ERROR)
        else:
            show_themed_message(self, "Failed to stage the book. Check logs for details.", "Staging Failed", wx.OK | wx.ICON_ERROR)

    def refresh_staging_tab(self):
        # Clear existing content
        for child in self.staging_tab_sizer.GetChildren():
            child.GetWindow().Destroy()

        from audiblez.database import get_staged_books_with_chapters, update_staged_chapter_selection, update_staged_book_final_compilation

        staged_books = get_staged_books_with_chapters()
        if not staged_books:
            no_books_label = wx.StaticText(self.staging_tab_panel, label="No books have been staged yet.")
            self.staging_tab_sizer.Add(no_books_label, 0, wx.ALL | wx.ALIGN_CENTER, 15)
        else:
            for book in staged_books:
                book_container = wx.Panel(self.staging_tab_panel, style=wx.BORDER_THEME)
                book_sizer = wx.BoxSizer(wx.VERTICAL)
                book_container.SetSizer(book_sizer)

                label = wx.StaticText(book_container, label=f"{book['title']} (Author: {book.get('author', 'N/A')})")
                font = label.GetFont()
                font.SetWeight(wx.FONTWEIGHT_BOLD)
                label.SetFont(font)
                book_sizer.Add(label, 0, wx.ALL, 5)

                # Final Compilation Checkbox for the book
                final_comp_checkbox = GenCheckBox(book_container, label="Enable Final Compilation for this Book")
                final_comp_checkbox.SetValue(book['final_compilation'])
                final_comp_checkbox.Bind(wx.EVT_CHECKBOX,
                                         lambda evt, b_id=book['id']:
                                         update_staged_book_final_compilation(b_id, evt.IsChecked()))
                book_sizer.Add(final_comp_checkbox, 0, wx.ALL | wx.ALIGN_LEFT, 5)

                # Chapters list for the book
                chapters_list_ctrl = None
                if book['chapters']:
                    agwStyle = ULC_REPORT
                    chapters_list_ctrl = UltimateListCtrl(book_container, agwStyle=agwStyle)
                    chapters_list_ctrl.InsertColumn(0, "Include", width=70)
                    chapters_list_ctrl.InsertColumn(1, "Chapter Title", width=200)  # Adjust width as needed
                    chapters_list_ctrl.InsertColumn(2, "Status", width=100)

                    for i, chap in enumerate(book['chapters']):
                        status_display = chap['status']
                        is_completed = chap['status'] == 'completed'
                        if is_completed:
                            status_display = "✅ Completed"

                        index = chapters_list_ctrl.InsertStringItem(i, "", it_kind=1)
                        chapters_list_ctrl.SetStringItem(index, 1, chap['title'])
                        chapters_list_ctrl.SetStringItem(index, 2, status_display)

                        if chap['is_selected_for_synthesis'] and not is_completed:
                            item = chapters_list_ctrl.GetItem(index)
                            item.Check(True)
                            chapters_list_ctrl.SetItem(item)

                        # Store chapter_id with the item for the event handler
                        chapters_list_ctrl.SetItemData(index, chap['id'])

                    # Define event handler for this specific chapters_list_ctrl
                    def create_chapter_check_handler(list_ctrl_instance, book_chapters_data):
                        def on_chapter_check(event):
                            chapter_idx = event.GetIndex()
                            chapter_id = list_ctrl_instance.GetItemData(chapter_idx)
                            # Find the chapter's original data to check its status
                            original_chapter_data = next((c for c in book_chapters_data if c['id'] == chapter_id), None)

                            if original_chapter_data and original_chapter_data['status'] == 'completed':
                                # If chapter is completed, prevent checking/unchecking by reverting the check state
                                item = list_ctrl_instance.GetItem(chapter_idx)
                                current_ui_checked_state = item.IsChecked()
                                item.Check(not current_ui_checked_state) # Revert
                                list_ctrl_instance.SetItem(item)
                                show_themed_message(self, "This chapter has already been processed and its selection cannot be changed here.",
                                              "Chapter Processed", wx.OK | wx.ICON_INFORMATION)
                                return

                            is_checked = list_ctrl_instance.IsItemChecked(chapter_idx)
                            update_staged_chapter_selection(chapter_id, is_checked)
                        return on_chapter_check

                    handler = create_chapter_check_handler(chapters_list_ctrl, book['chapters'])
                    chapters_list_ctrl.Bind(wx.EVT_LIST_ITEM_CHECKED, handler)
                    chapters_list_ctrl.Bind(wx.EVT_LIST_ITEM_UNCHECKED, handler) # Same handler for uncheck
                    book_sizer.Add(chapters_list_ctrl, 1, wx.ALL | wx.EXPAND, 5)
                else:
                    no_chapters_label = wx.StaticText(book_container, label="This book has no chapters.")
                    book_sizer.Add(no_chapters_label, 0, wx.ALL, 5)

                # Add "Queue Selected Chapters" button for this book
                queue_selected_button = wx.Button(book_container, label="▶️ Queue Selected Chapters")
                queue_selected_button.Bind(wx.EVT_BUTTON, lambda evt, b_id=book['id'], b_title=book['title'], list_ctrl=chapters_list_ctrl: self.on_queue_selected_staged_chapters(evt, b_id, b_title, list_ctrl))
                book_sizer.Add(queue_selected_button, 0, wx.ALL | wx.ALIGN_CENTER, 10)

                self.staging_tab_sizer.Add(book_container, 0, wx.ALL | wx.EXPAND, 10)

        self.staging_tab_panel.SetupScrolling()
        self.staging_tab_panel.Layout()
        # self.Layout() # Main frame layout, might be too broad, staging_tab_panel.Layout() should suffice.
        self.splitter.Layout() # Layout the main splitter that contains left and right
        self.Layout() # Full frame layout might be ineeded if sizers changed overall frame size.
        self.apply_theme(self.theme_name)

    def update_staging_tab_for_processed_chapters(self, processed_staged_chapter_ids: list[int]):
        """
        Refreshes the staging tab to reflect the 'completed' status of chapters
        that were processed as part of a queue item.
        """
        # This method will find the relevant book and chapter in the staging tab UI
        # and update its visual representation (e.g., disable checkbox, show checkmark).
        # For now, a full refresh of the staging tab is the simplest way.
        # More granular updates can be implemented if performance becomes an issue.
        print(f"Updating staging tab for processed chapter IDs: {processed_staged_chapter_ids}")
        # Potentially, find the specific book and chapters_list_ctrl to update
        # For now, just trigger a full refresh.
        self.refresh_staging_tab()


    def on_queue_selected_staged_chapters(self, event, book_id, book_title, chapters_list_ctrl):
        raw_selected_chapters_data = []
        if not chapters_list_ctrl: # Should not happen if button is present
            show_themed_message(self, "Error: Chapter list not found for this book.", "Error", wx.OK | wx.ICON_ERROR)
            return

        for i in range(chapters_list_ctrl.GetItemCount()):
            if chapters_list_ctrl.IsItemChecked(i):
                chapter_id_in_db = chapters_list_ctrl.GetItemData(i) # This is the DB ID of the chapter
                chapter_title = chapters_list_ctrl.GetItem(i, 1).GetText()
                text = db.get_chapter_text_content(chapter_id_in_db)
                if text is None:
                    # Log warning, but allow queuing. process_next_queue_item will try to re-fetch.
                    print(f"Warning: Could not fetch text for staged chapter ID {chapter_id_in_db} ('{chapter_title}') during queuing. Will attempt fetch during processing.")

                raw_selected_chapters_data.append({
                    'db_id': chapter_id_in_db, # Original DB ID from staged_chapters
                    'title': chapter_title,
                    'text_content': text # Store fetched text (might be None if fetch failed)
                })

        if not raw_selected_chapters_data:
            show_themed_message(self, "No chapters selected. Please check the selection.",
                          "No Selection", wx.OK | wx.ICON_INFORMATION)
            return

        # Prepare the 'chapters' list for add_item_to_queue
        final_chapters_for_db = []
        for idx, chap_data in enumerate(raw_selected_chapters_data):
            final_chapters_for_db.append({
                'staged_chapter_id': chap_data['db_id'], # This key is expected by add_item_to_queue
                'title': chap_data['title'],
                'text_content': chap_data['text_content'],
                'order': idx # This provides the chapter_order for queued_chapters
            })

        # Retrieve current global synthesis settings
        current_engine = 'cuda' if self.cuda_toggle.GetValue() else 'cpu'
        current_output_folder = self.output_folder_text_ctrl.GetValue()

        synthesis_settings = { # This is a dict
            'engine': current_engine,
            'output_folder': current_output_folder,
            'm4b_assembly_method': self.m4b_assembly_method,
        }

        db_queue_details = {
            'staged_book_id': book_id,
            'book_title': book_title,
            'source_path': None, # Staged items don't have a direct source_path for the queue item itself
            'synthesis_settings': synthesis_settings,
            'chapters': final_chapters_for_db
        }

        # print(f"DEBUG: Before DB reload, self.queue_items: {self.queue_items}")
        new_item_id = db.add_item_to_queue(db_queue_details)
        if new_item_id:
            self.queue_items = db.get_queued_items() # Reload queue
            # print(f"DEBUG: After DB reload, self.queue_items: {self.queue_items}")
            # print("DEBUG: Calling refresh_queue_tab()")
            self.refresh_queue_tab()
            self.notebook.SetSelection(self.notebook.GetPageCount() - 1)
            show_themed_message(self, f"Added '{book_title}' (with {len(final_chapters_for_db)} selected chapter(s)) to the queue.",
                          "Added to Queue", wx.OK | wx.ICON_INFORMATION)
        else:
            show_themed_message(self, "Failed to add item to the database queue.", "Error", wx.OK | wx.ICON_ERROR)


    def on_tidy_text(self, event):
        """Clean up text formatting by removing excessive line breaks and improving readability"""
        if not self.selected_chapter:
            show_themed_message(self, "No chapter selected. Please select a chapter first.", 
                              "No Chapter Selected", wx.OK | wx.ICON_INFORMATION)
            return
        
        original_text = self.text_area.GetValue()
        if not original_text.strip():
            show_themed_message(self, "The selected chapter appears to be empty.", 
                              "Empty Chapter", wx.OK | wx.ICON_INFORMATION)
            return
        
        # Apply text cleaning algorithms
        cleaned_text = self._clean_epub_text(original_text)
        
        # Update the text area and chapter data
        self.text_area.SetValue(cleaned_text)
        self.selected_chapter.extracted_text = cleaned_text
        
        show_themed_message(self, "Text formatting has been cleaned up successfully!", 
                          "Text Tidied", wx.OK | wx.ICON_INFORMATION)
    
    def _clean_epub_text(self, text):
        """Apply various text cleaning algorithms to improve formatting"""
        import re
        
        # Initial cleanup - remove excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r' *\n *', '\n', text)  # Remove spaces around newlines
        
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []
        
        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            
            # Handle empty lines - preserve single empty lines, remove excessive ones
            if not current_line:
                # Only add empty line if the last added line wasn't empty
                if cleaned_lines and cleaned_lines[-1] != '':
                    cleaned_lines.append('')
                i += 1
                continue
            
            # Look ahead to see if we should merge with subsequent lines
            merged_line = current_line
            j = i + 1
            
            while j < len(lines):
                next_line = lines[j].strip()
                
                # Skip empty lines when looking ahead
                if not next_line:
                    j += 1
                    continue
                
                # Determine if current line should be merged with next line
                should_merge = False
                
                # Case 1: Current line doesn't end with sentence terminator and isn't a dialogue/quote line
                if (not re.search(r'[.!?:;]$', merged_line) and 
                    not merged_line.endswith('"') and 
                    not merged_line.endswith("'") and
                    len(merged_line) > 5):
                    should_merge = True
                
                # Case 2: Current line ends with comma, dash, or continuation punctuation
                elif re.search(r'[,\-—]$', merged_line):
                    should_merge = True
                
                # Case 3: Next line starts with lowercase (clear continuation)
                elif (next_line and next_line[0].islower() and 
                      not next_line.startswith('"') and 
                      not next_line.startswith("'")):
                    should_merge = True
                
                # Case 4: Current line ends with word + next starts with word (broken word wrapping)
                elif (re.search(r'\w$', merged_line) and 
                      re.search(r'^\w', next_line) and
                      not next_line[0].isupper()):
                    should_merge = True
                
                # Don't merge if next line looks like a new paragraph/section
                if (should_merge and
                    (re.search(r'^(Chapter|CHAPTER|\d+\.|\*)', next_line) or
                     len(next_line) < 3 or  # Very short lines might be intentional breaks
                     next_line.isupper())):  # All caps might be headings
                    should_merge = False
                
                if should_merge:
                    # Merge the lines with appropriate spacing
                    if merged_line.endswith('-') or merged_line.endswith('—'):
                        # Hyphenated word continuation
                        merged_line = merged_line[:-1] + next_line
                    else:
                        # Normal space between words
                        merged_line += ' ' + next_line
                    j += 1
                else:
                    break
            
            cleaned_lines.append(merged_line)
            i = j if j > i + 1 else i + 1
        
        # Rejoin and apply final formatting
        text = '\n'.join(cleaned_lines)
        
        # Fix punctuation spacing issues
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Space after sentence endings
        text = re.sub(r'([.!?])\s*\n', r'\1\n', text)  # Clean line endings after sentences
        
        # Normalize paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean up paragraph breaks
        
        # Add natural spacing after sentences within paragraphs
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Ensure space between sentences
        
        # Final cleanup
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces on lines
        text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces on lines
        text = text.strip()
        
        return text

    def on_preview_chapter(self, event):
        import numpy as np
        import soundfile
        from tempfile import NamedTemporaryFile
        from audiblez import core
        from chatterbox import ChatterboxTTS
        import subprocess
        button = event.GetEventObject()
        button.SetLabel("⏳")
        button.Disable()

        def generate_preview():
            core.load_spacy()
            text = self.selected_chapter.extracted_text[:300] if self.selected_chapter else ""
            if len(text) == 0:
                button.SetLabel("🔊 Preview")
                button.Enable()
                return
            # Gather Chatterbox model config from UI controls
            tts_model_config = {k: ctrl.GetValue() for k, ctrl in self.model_config_controls.items()}
            voice_clone_sample = self.clone_sample_path.GetValue() or None
            # Chatterbox-TTS: Use from_pretrained and .generate for preview
            model = ChatterboxTTS.from_pretrained('cuda' if torch.cuda.is_available() else 'cpu')
            wav = model.generate(
                text,
                audio_prompt_path=voice_clone_sample,
                exaggeration=tts_model_config.get('exaggeration', 1.0),
                temperature=tts_model_config.get('temperature', 1.0),
                cfg_weight=tts_model_config.get('cfg_weight', 5.0),
                min_p=tts_model_config.get('min_p', 0.05),
                top_p=tts_model_config.get('top_p', 0.95),
                repetition_penalty=tts_model_config.get('repetition_penalty', 1.0),
            )
            tmp_preview_wav_file = NamedTemporaryFile(suffix='.wav', delete=False)
            if hasattr(wav, 'cpu'):  # PyTorch tensor
                audio_data = wav.squeeze(0).cpu().numpy()
            else:  # numpy array
                audio_data = wav.squeeze(0) if wav.ndim > 1 else wav
            
            # Remove excessive pauses from preview audio as well
            audio_data = core.remove_excessive_pauses(audio_data, model.sr, max_pause_duration=3.0)
            
            soundfile.write(tmp_preview_wav_file, audio_data, model.sr)
            cmd = ['ffplay', '-autoexit', '-nodisp', tmp_preview_wav_file.name]
            subprocess.run(cmd)
            button.SetLabel("🔊 Preview")
            button.Enable()

        if len(self.preview_threads) > 0:
            for thread in self.preview_threads:
                thread.join()
            self.preview_threads = []
        thread = threading.Thread(target=generate_preview)
        thread.start()
        self.preview_threads.append(thread)

    def on_start(self, event):
        if platform.system() == "macOS":
            import caffeine
            caffeine.on(display=False)  # Prevent sleep during synthesis
        self.synthesis_in_progress = True
        file_path = self.selected_file_path
        selected_chapters = []
        for i in range(self.table.GetItemCount()):
            if self.table.IsItemChecked(i):
                selected_chapters.append(self.document_chapters[i])
        self.start_button.Disable()
        self.params_panel.Disable()

        # self.table.EnableCheckBoxes(False) # Not available in UltimateListCtrl
        for chapter_index, chapter in enumerate(self.document_chapters): # document_chapters could be from EPUB or Calibre
            if chapter in selected_chapters:
                self.set_table_chapter_status(chapter_index, "Planned")
                # self.table.SetItem(chapter_index, 0, '✔️') # Checkmarking handled by table.CheckItem in create_chapters_table_panel

        # Gather TTS model config and voice clone sample
        tts_model_config = {k: ctrl.GetValue() for k, ctrl in self.model_config_controls.items()}
        voice_clone_sample = self.clone_sample_path.GetValue() or None
        core_params = {
            'file_path': file_path, # This is the original input file path
            'pick_manually': False,
            'output_folder': self.output_folder_text_ctrl.GetValue(),
            'selected_chapters': selected_chapters,
            'calibre_metadata': None, # Default to None
            'calibre_cover_image_path': None, # Default to None
            'm4b_assembly_method': self.m4b_assembly_method,
            'tts_model_config': tts_model_config,
            'voice_clone_sample': voice_clone_sample
        }

        # Check if this book was loaded via Calibre by inspecting self.book_data
        # self.book_data is set in on_open_with_calibre
        if hasattr(self, 'book_data') and self.book_data and 'metadata' in self.book_data and 'cover_image_path' in self.book_data:
            # This indicates a Calibre-loaded book currently in the UI
            # We need to ensure this self.book_data corresponds to the file_path being processed.
            # For on_start, self.selected_file_path IS file_path.
            if self.selected_file_path == file_path:
                print("Passing Calibre-derived metadata and cover path to core.main for single synthesis.")
                core_params['calibre_metadata'] = self.book_data['metadata']
                core_params['calibre_cover_image_path'] = self.book_data['cover_image_path']
            else:
                # This case should ideally not happen if UI state is consistent.
                print("Warning: Mismatch between current file_path and selected_file_path for Calibre data in on_start. Proceeding without Calibre specifics.")


        print('Starting Audiobook Synthesis', core_params)
        self.core_thread = CoreThread(params=core_params)
        self.core_thread.start()

    def on_open(self, event):
        result, file_path = show_themed_file_dialog(self, "Open EPUB File", wildcard="*.epub", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if result != wx.ID_OK or not file_path:
            return
        print(f"Selected file: {file_path}")
        if self.synthesis_in_progress:
            show_themed_message(self, "Audiobook synthesis is still in progress. Please wait for it to finish.", "Audiobook Synthesis in Progress")
        else:
            wx.CallAfter(self._load_epub_file, file_path)

    def on_open_with_calibre(self, event):
        from audiblez.core import get_calibre_ebook_convert_path, convert_ebook_with_calibre, extract_chapters_and_metadata_from_calibre_html
        import tempfile
        import shutil

        # Define the GUI callback that will be passed to the core function.
        # This callback is only invoked if the core function cannot find Calibre automatically.
        def ask_user_for_calibre_path_gui():
            # First, show an informational dialog.
            info_message = (
                "Audiblez needs to know where Calibre is installed to convert this book format.\n\n"
                "Please locate the 'ebook-convert' program inside your Calibre installation folder.\n\n"
                "What to look for:\n"
                "- On Windows, this is often 'C:\\Program Files\\Calibre2\\ebook-convert.exe'.\n"
                "- On macOS, this is usually in '/Applications/calibre.app/Contents/MacOS/ebook-convert'.\n\n"
                "The folder containing this file should also have 'calibre-debug'."
            )
            dialog = wx.MessageDialog(self, info_message, "Locate Calibre Program", wx.OK | wx.CANCEL | wx.ICON_INFORMATION)
            response = dialog.ShowModal()
            dialog.Destroy()

            if response == wx.ID_CANCEL:
                return None # User cancelled the info dialog

            # If user clicks OK, show the file picker dialog.
            message = "Select the 'ebook-convert' executable"
            if platform.system() == "Windows":
                wildcard = "ebook-convert executable (ebook-convert.exe)|ebook-convert.exe|All files (*.*)|*.*"
            else:
                wildcard = "ebook-convert executable (ebook-convert)|ebook-convert|All files (*.*)|*.*"

            result, selected_file = show_themed_file_dialog(self, message, wildcard=wildcard,
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
            if result == wx.ID_CANCEL or not selected_file:
                return None  # User cancelled
            
            # Return the directory containing the selected file, as the core function expects.
            selected_path = Path(selected_file)
            return str(selected_path.parent)

        # 1. Check for Calibre's existence first, prompting the user if necessary.
        # We pass our GUI callback function here.
        ebook_convert_exe = get_calibre_ebook_convert_path(ui_callback_for_path_selection=ask_user_for_calibre_path_gui)

        # If no path was found (either automatically or by the user), abort.
        if not ebook_convert_exe:
            show_themed_message(self, "Calibre's 'ebook-convert' tool could not be found. The process has been cancelled.",
                          "Calibre Not Found", wx.OK | wx.ICON_ERROR)
            return

        # 2. If Calibre is found, now prompt the user to select an ebook file.
        wildcard_str = "Ebook files (*.epub;*.mobi;*.azw;*.azw3;*.fb2;*.lit;*.pdf)|*.epub;*.mobi;*.azw;*.azw3;*.fb2;*.lit;*.pdf|All files (*.*)|*.*"
        result, input_ebook_path = show_themed_file_dialog(self, "Select Ebook to Convert with Calibre", wildcard=wildcard_str,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if result == wx.ID_CANCEL or not input_ebook_path:
            return # User cancelled file selection

        if self.synthesis_in_progress:
            show_themed_message(self, "Audiobook synthesis is in progress. Please wait for it to finish.",
                          "Synthesis Busy", wx.OK | wx.ICON_WARNING)
            return

        # 3. Proceed with the conversion and UI update logic.
        temp_html_output_dir = tempfile.mkdtemp(prefix="audiblez_calibre_")
        print(f"Temporary directory for Calibre HTML output: {temp_html_output_dir}")

        try:
            wx.BeginBusyCursor()
            # We already have the path, so we don't need to pass the callback to convert_ebook_with_calibre.
            # It will call get_calibre_ebook_convert_path again, but it will find it in the DB or PATH now.
            html_file_path, opf_file_path, cover_image_path = convert_ebook_with_calibre(
                input_ebook_path,
                temp_html_output_dir,
                ui_callback_for_path_selection=None # Path is already found and saved.
            )
            wx.EndBusyCursor()

            if not html_file_path:
                show_themed_message(self, f"Failed to convert '{Path(input_ebook_path).name}' using Calibre. Check console for errors.",
                              "Calibre Conversion Failed", wx.OK | wx.ICON_ERROR)
                return

            wx.BeginBusyCursor()
            extracted_chapters, book_metadata = extract_chapters_and_metadata_from_calibre_html(html_file_path, opf_file_path)
            wx.EndBusyCursor()

            if not extracted_chapters:
                title_from_meta = book_metadata.get('title', Path(input_ebook_path).stem)
                show_themed_message(self, f"Could not extract chapters from the HTML output of '{title_from_meta}'. The book might be empty or in an unexpected format.",
                              "Chapter Extraction Failed", wx.OK | wx.ICON_WARNING)

            # Store calibre-specific data that might be used by other functions
            self.book_data = {
                'cover_image_path': cover_image_path,
                'metadata': book_metadata
            }
            
            cover_info = {'type': 'path', 'content': cover_image_path} if cover_image_path else None

            self._load_book_data_into_ui(
                book_title=book_metadata.get('title', Path(input_ebook_path).stem),
                book_author=book_metadata.get('creator', "Unknown Author"),
                document_chapters=extracted_chapters,
                source_path=input_ebook_path,
                book_object=None, # No epub object for calibre
                cover_info=cover_info
            )

            if extracted_chapters:
                show_themed_message(self, f"Successfully processed '{self.selected_book_title}' using Calibre.",
                              "Processing Complete", wx.OK | wx.ICON_INFORMATION)

        finally:
            if Path(temp_html_output_dir).exists():
                try:
                    shutil.rmtree(temp_html_output_dir)
                    print(f"Cleaned up temporary directory: {temp_html_output_dir}")
                except Exception as e:
                    print(f"Error cleaning up temporary directory {temp_html_output_dir}: {e}")
            if wx.IsBusy():
                wx.EndBusyCursor()


    def on_exit(self, event):
        self.Close()

    def set_table_chapter_status(self, chapter_index, status):
        if hasattr(self, 'table') and self.table:
            self.table.SetStringItem(chapter_index, 3, status)

    def open_folder_with_explorer(self, folder_path):
        try:
            if platform.system() == 'Windows':
                subprocess.Popen(['explorer', folder_path])
            elif platform.system() == 'Linux':
                subprocess.Popen(['xdg-open', folder_path])
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', folder_path])
        except Exception as e:
            print(e)


class CoreThread(threading.Thread):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        from audiblez import core
        core.main(**self.params, post_event=self.post_event)

    def post_event(self, event_name, **kwargs):
        # eg. 'EVENT_CORE_PROGRESS' -> EventCoreProgress, EVENT_CORE_PROGRESS
        EventObject, EVENT_CODE = EVENTS[event_name]
        event_object = EventObject()
        for k, v in kwargs.items():
            setattr(event_object, k, v)
        wx.PostEvent(wx.GetApp().GetTopWindow(), event_object)


class ScheduleDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="Schedule Queue", size=(400, 250))
        self.selected_datetime = None

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Instruction
        instruction = wx.StaticText(panel, label="Select a date and time for the queue to start:")
        vbox.Add(instruction, 0, wx.ALL | wx.EXPAND, 15)

        # Date Picker
        date_box = wx.BoxSizer(wx.HORIZONTAL)
        date_label = wx.StaticText(panel, label="Date:")
        self.date_picker = wx.adv.CalendarCtrl(panel, -1)

        # Initialize with current date or saved schedule
        current_schedule_ts = db.load_schedule_time()
        initial_date = datetime.now()
        if current_schedule_ts and current_schedule_ts > 0:
            try:
                initial_date = datetime.fromtimestamp(current_schedule_ts)
            except ValueError: # Handle potential invalid timestamp from DB
                pass
        self.date_picker.SetDate(wx.DateTime.FromDMY(initial_date.day, initial_date.month - 1, initial_date.year))

        date_box.Add(date_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        date_box.Add(self.date_picker, 1, wx.EXPAND)
        vbox.Add(date_box, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        # Time Picker
        time_box = wx.BoxSizer(wx.HORIZONTAL)
        time_label = wx.StaticText(panel, label="Time (HH:MM):")
        # Using TimePickerCtrl if available and suitable, otherwise TextCtrl
        # For simplicity and wider compatibility, using TextCtrl with format hint
        self.time_picker = wx.TextCtrl(panel, value=initial_date.strftime("%H:%M"))
        self.time_picker.SetToolTip("Enter time in 24-hour HH:MM format")

        time_box.Add(time_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        time_box.Add(self.time_picker, 1, wx.EXPAND)
        vbox.Add(time_box, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        # Buttons
        hbox_buttons = wx.BoxSizer(wx.HORIZONTAL)
        ok_button = wx.Button(panel, label="Set Schedule", id=wx.ID_OK)
        ok_button.SetDefault()
        clear_button = wx.Button(panel, label="Clear Schedule")
        cancel_button = wx.Button(panel, label="Cancel", id=wx.ID_CANCEL)

        ok_button.Bind(wx.EVT_BUTTON, self.on_ok)
        clear_button.Bind(wx.EVT_BUTTON, self.on_clear)
        # cancel_button's ID_CANCEL is handled by default dialog behavior

        hbox_buttons.Add(ok_button)
        hbox_buttons.Add(clear_button, 0, wx.LEFT, 5)
        hbox_buttons.Add(cancel_button, 0, wx.LEFT, 5)
        vbox.Add(hbox_buttons, 0, wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, 10)

        panel.SetSizer(vbox)
        self.CentreOnParent()

        # Apply theme to the dialog
        panel.SetBackgroundColour(theme['background'])
        vbox.SetBackgroundColour(theme['background'])
        instruction.SetForegroundColour(theme['text'])
        date_label.SetForegroundColour(theme['text'])
        time_label.SetForegroundColour(theme['text'])

        # Theme controls
        self.time_picker.SetBackgroundColour(theme['panel'])
        self.time_picker.SetForegroundColour(theme['text'])
        ok_button.SetBackgroundColour(theme['button_face'])
        ok_button.SetForegroundColour(theme['button_text'])
        clear_button.SetBackgroundColour(theme['button_face'])
        clear_button.SetForegroundColour(theme['button_text'])
        cancel_button.SetBackgroundColour(theme['button_face'])
        cancel_button.SetForegroundColour(theme['button_text'])

        # Theme Calendar
        self.date_picker.SetBackgroundColour(theme['panel'])
        self.date_picker.SetForegroundColour(theme['text'])
        self.date_picker.SetHeaderColours(theme['highlight'], theme['highlight_text'])
        self.date_picker.SetHighlightColours(theme['highlight'], theme['highlight_text'])

        # Apply Windows dark mode if available
        if theme == palettes['dark']:
            apply_windows_dark_mode_to_window(self, True)

    def on_ok(self, event):
        wx_date = self.date_picker.GetDate()
        date_val = datetime(wx_date.GetYear(), wx_date.GetMonth() + 1, wx_date.GetDay())

        time_str = self.time_picker.GetValue()
        try:
            time_val = datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            show_themed_message(self, "Invalid time format. Please use HH:MM (24-hour).", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.selected_datetime = datetime.combine(date_val.date(), time_val)
        self.EndModal(wx.ID_OK)

    def on_clear(self, event):
        self.selected_datetime = None # Indicate clearance
        self.EndModal(wx.ID_OK) # Still ID_OK to signal dialog was actioned, parent checks selected_datetime

    def get_selected_datetime(self):
        return self.selected_datetime


def initialize_palettes():
    """
    Initializes the color palettes that depend on wx.SystemSettings.
    This must be called after the wx.App object has been created.
    """
    palettes['light']['button_face'] = wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE)
    palettes['light']['button_text'] = wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNTEXT)
    palettes['light']['list_header'] = wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE)


def main():
    print('Starting GUI...')
    app = wx.App(False)
    initialize_palettes()  # Initialize colors after app creation
    frame = MainWindow(None, "Audiblez - Generate Audiobooks from E-books")
    frame.Show(True)
    frame.Layout()
    app.SetTopWindow(frame)
    print('Done.')
    app.MainLoop()


if __name__ == '__main__':
    main()
