import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import os


class ImageSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Selector")
        self.root.geometry("800x800")

        self.selected_images = []
        self.image_refs = []  # Keep references to image objects

        # Create the frames
        self.left_frame = ttk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = ttk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.top_right_frame = ttk.Frame(self.right_frame)
        self.top_right_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        self.bottom_right_frame = ttk.Frame(self.right_frame)
        self.bottom_right_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Create a Treeview widget for folder structure
        self.tree = ttk.Treeview(self.left_frame)
        self.tree.pack(expand=True, fill=tk.Y)
        self.tree.bind("<Double-1>", self.on_tree_double_click)

        # Create a canvas to display images
        self.canvas = tk.Canvas(self.top_right_frame, bg="white")
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<Motion>", self.on_canvas_hover)

        # Initialize the root node
        self.root_node = self.tree.insert("", tk.END, text="Root", open=True)

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.populate_tree(self.root_node, script_dir, depth=0, max_depth=3)

        # Keep track of delete buttons
        self.delete_buttons = []

        # Create chat interface
        self.chat_log = scrolledtext.ScrolledText(
            self.bottom_right_frame, state='disabled', height=10)
        self.chat_log.pack(side=tk.TOP, fill=tk.X)

        self.chat_entry = tk.Entry(self.bottom_right_frame)
        self.chat_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.chat_button = tk.Button(
            self.bottom_right_frame, text="Send", command=self.send_message)
        self.chat_button.pack(side=tk.LEFT)

        self.chat_role = tk.StringVar(value="A")
        self.role_a_button = tk.Radiobutton(
            self.bottom_right_frame, text="A", variable=self.chat_role, value="A")
        self.role_b_button = tk.Radiobutton(
            self.bottom_right_frame, text="B", variable=self.chat_role, value="B")
        self.role_a_button.pack(side=tk.LEFT)
        self.role_b_button.pack(side=tk.LEFT)

        # Add Clear Chat and Submit buttons
        self.clear_button = tk.Button(
            self.bottom_right_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.submit_button = tk.Button(
            self.bottom_right_frame, text="Submit", command=self.submit)
        self.submit_button.pack(side=tk.LEFT, padx=5)

    def populate_tree(self, parent, path, depth, max_depth):
        if depth > max_depth:
            return
        try:
            for item in os.listdir(path):
                abs_path = os.path.join(path, item)
                node = self.tree.insert(parent, tk.END, text=item, open=False)
                if os.path.isdir(abs_path) and not os.path.islink(abs_path):
                    self.populate_tree(node, abs_path, depth + 1, max_depth)
        except PermissionError:
            pass

    def on_tree_double_click(self, event):
        selected_item = self.tree.selection()[0]
        path = self.get_full_path(selected_item)
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            if len(self.selected_images) >= 3:
                messagebox.showerror(
                    "Error", "You can select up to 3 images only.")
                return
            self.selected_images.append(path)
            self.display_images()
        else:
            messagebox.showerror("Error", "Please select an image file.")

    def get_full_path(self, node):
        path = []
        while node != self.root_node:
            path.append(self.tree.item(node)["text"])
            node = self.tree.parent(node)
        path.reverse()
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), *path)

    def display_images(self):
        self.canvas.delete("all")
        self.delete_buttons.clear()
        self.image_refs.clear()  # Clear previous image references

        x, y = 10, 10
        max_width = self.canvas.winfo_width() // 5
        max_height = self.canvas.winfo_height() // 2
        y_offset = max_height + 5  # Reduce vertical space between rows

        for idx, img_path in enumerate(self.selected_images):
            img = Image.open(img_path)
            img.thumbnail((max_width, max_height))
            img_tk = ImageTk.PhotoImage(img)

            item_id = self.canvas.create_image(
                x, y, anchor=tk.NW, image=img_tk)
            # Keep a reference to the image object
            self.image_refs.append(img_tk)

            img_width, img_height = img.size
            button = tk.Button(self.canvas, text="X",
                               command=lambda idx=idx: self.remove_image(idx))
            self.delete_buttons.append(
                (button, x, y, x + img_width, y + img_height))

            self.canvas.create_window(
                x + img_width - 15, y + 15, window=button)
            button.lower()

            x += max_width + 10
            if x + max_width > self.canvas.winfo_width():
                x = 10
                y += y_offset

    def on_canvas_hover(self, event):
        x, y = event.x, event.y
        for button, bx1, by1, bx2, by2 in self.delete_buttons:
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                button.lift()
            else:
                button.lower()

    def remove_image(self, idx):
        del self.selected_images[idx]
        self.display_images()

    def send_message(self):
        message = self.chat_entry.get()
        if message:
            role = self.chat_role.get()
            self.chat_log.config(state='normal')
            self.chat_log.insert(tk.END, f"{role}: {message}\n")
            self.chat_log.config(state='disabled')
            self.chat_log.yview(tk.END)
            self.chat_entry.delete(0, tk.END)

    def clear_chat(self):
        self.chat_log.config(state='normal')
        self.chat_log.delete(1.0, tk.END)
        self.chat_log.config(state='disabled')

    def submit(self):
        chat_content = self.chat_log.get(1.0, tk.END).strip()
        submission = {
            "images": self.selected_images,
            "chat": chat_content
        }
        print(submission)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSelectorApp(root)
    root.mainloop()
