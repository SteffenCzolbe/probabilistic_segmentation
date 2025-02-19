import torch
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image


def image_to_numpy(tensor):
    """
    transforms a 2d or 3d torch.tensor to a numpy array.
    C x H x W x D becomes H x W x D x C
    1 x H x W x D becomes H x W x D
    C x H x W becomes H x W x C
    1 x H x W becomes H x W
    """
    # to numpy
    array = tensor.detach().cpu().numpy()
    # channel to back
    permuted = np.moveaxis(array, 0, -1)
    # remove channel of size 1
    if permuted.shape[-1] == 1:
        permuted = permuted[..., 0]
    return permuted


class Fig:
    def __init__(self, rows=1, cols=1, title=None, figsize=None, background=False):
        """
        instantiates a plot.
        Parameters:
            rows: how many plots per row
            cols: how many plots per column
            title: title of the figure
        """
        # instantiate plot
        self.fig, self.axs = plt.subplots(
            nrows=rows, ncols=cols, dpi=300, figsize=figsize, frameon=background
        )

        # set title
        self.fig.suptitle(title)

        # extend empty dimensions to array
        if cols == 1 and rows == 1:
            self.axs = np.array([[self.axs]])
        elif cols == 1:
            self.axs = np.array([[ax] for ax in self.axs])
        elif rows == 1:
            self.axs = np.array([self.axs])

        # hide all axis
        for row in self.axs:
            for ax in row:
                ax.axis("off")

    def plot_img(self, row, col, image, title=None, vmin=None, vmax=None):
        """
        plots a tensor of the form C x H x W at position row, col.
        C needs to be either C=1 or C=3
        Parameters:
            row: the row to plot the image
            col: the clolumn to plot the image
            image: Tensor of shape C x H x W
            title: optional title
            vmin: optinal lower bound for color scaling
            vmax: optional higher bound for color scaling
        """
        # convert to numpy
        img = image_to_numpy(image)
        if len(img.shape) == 2:
            # plot greyscale image
            self.axs[row, col].imshow(
                img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="none"
            )
        elif len(img.shape) == 3:
            # last channel is color channel
            self.axs[row, col].imshow(img)
        self.axs[row, col].set_aspect("equal")
        self.axs[row, col].title.set_text(title)
        return self

    def plot_contour(
        self, row, col, mask, contour_class, width=3, rgba=(36, 255, 12, 255)
    ):
        """
        imposes a contour-line overlay onto a plot
        Parameters:
            row: the row to plot the image
            col: the clolumn to plot the image
            mask: Tensor of shape C x H x W
            contour_class: the class to make a contour of
            width: thickness of contours.
            rgba: color of the contour lines. RGB or RGBA formats
        """
        # convert to numpy
        mask = image_to_numpy(mask)

        # get mask
        mask = mask == contour_class

        if len(rgba) == 3:
            # add alpha-channel
            rgba = (*rgba, 255)

        # find countours
        import cv2

        outline = np.zeros((*mask.shape[:2], 4), dtype=np.uint8) * 255
        cnts = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(outline, [c], -1, rgba, thickness=width)
        self.axs[row, col].imshow(
            outline.astype(np.float) / 255,
            vmin=0,
            vmax=1,
            interpolation="none",
            alpha=1.0,
        )

        return self

    def plot_overlay_class_mask(
        self, row, col, class_mask, num_classes, colors, alpha=0.4
    ):
        """
        imposes a color-coded class_mask onto a plot
        class_mask needs to be of the form 1 x H x W of type long.
        Parameters:
            row: the row to plot the image
            col: the clolumn to plot the image
            class_mask: Tensor of shape 1 x H x W
            num_classes: number of classes
            colors: list of colors to plot. RGB. or RGBA eg: [(0,0,255)]
            alpha: alpha-visibility of the overlay. Default 0.4
        """
        # one-hot encode the classes
        class_masks = (
            torch.nn.functional.one_hot(
                class_mask[0].long(), num_classes=num_classes)
            .detach()
            .cpu()
            .numpy()
        )
        img = np.zeros((*class_masks.shape[:2], 4))
        for c in range(num_classes):
            color = colors[c]
            if color is None:
                color = (0, 0, 0, 0)
            if len(color) == 3:
                color = (*color, 255)
            img[class_masks[:, :, c] == 1] = np.array(color) / 255
        # back to tensor for the next function
        img = torch.tensor(img).permute(-1, 0, 1)
        self.plot_overlay(row, col, img, alpha=alpha)

    def plot_overlay(self, row, col, mask, alpha=0.4, vmin=None, vmax=None, cmap='jet', colorbar=False, colorbar_label=None):
        """
        imposes an overlay onto a plot
        Overlay needs to be of the form C x H x W
        C needs to be either C=1 or C=3 or C=4
        Parameters:
            row: the row to plot the image
            col: the clolumn to plot the image
            mask: Tensor of shape C x H x W
            alpha: alpha-visibility of the overlay. Default 0.4
            vmin: optinal lower bound for color scaling
            vmax: optional higher bound for color scaling
        """
        # Add color bar, as per https://github.com/matplotlib/matplotlib/issues/15010
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.transforms import Bbox
        from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

        class RemainderFixed(axes_size.Scaled):
            def __init__(self, xsizes, ysizes, divider):
                self.xsizes = xsizes
                self.ysizes = ysizes
                self.div = divider

            def get_size(self, renderer):
                xrel, xabs = axes_size.AddList(
                    self.xsizes).get_size(renderer)
                yrel, yabs = axes_size.AddList(
                    self.ysizes).get_size(renderer)
                bb = Bbox.from_bounds(
                    *self.div.get_position()).transformed(self.div._fig.transFigure)
                w = bb.width/self.div._fig.dpi - xabs
                h = bb.height/self.div._fig.dpi - yabs
                return 0, min([w, h])

        def make_square_axes_with_colorbar(ax, size=0.1, pad=0.1):
            """ Make an axes square, add a colorbar axes next to it, 
                Parameters: size: Size of colorbar axes in inches
                            pad : Padding between axes and cbar in inches
                Returns: colorbar axes
            """
            divider = make_axes_locatable(ax)
            margin_size = axes_size.Fixed(size)
            pad_size = axes_size.Fixed(pad)
            xsizes = [pad_size, margin_size]
            yhax = divider.append_axes(
                "right", size=margin_size, pad=pad_size)
            divider.set_horizontal(
                [RemainderFixed(xsizes, [], divider)] + xsizes)
            divider.set_vertical([RemainderFixed(xsizes, [], divider)])
            return yhax

        # convert to numpy
        mask = image_to_numpy(mask)
        if len(mask.shape) == 2:
            # plot greyscale image
            im = self.axs[row, col].imshow(
                mask,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="none",
                alpha=alpha,
            )
        elif len(mask.shape) in [3, 4]:
            # last channel is color channel
            im = self.axs[row, col].imshow(mask, alpha=alpha)

        if colorbar:
            cax = make_square_axes_with_colorbar(
                self.axs[row, col], size=0.1, pad=0.1)
            cbar = self.fig.colorbar(im, cax=cax)
            cbar.ax.set_ylabel(colorbar_label)

        return self

    def show(self):
        plt.show()

    def save(self, path, close=True):
        """
        saves the current figure.
        Parameters:
            path: path to save at. Including extension. eg. '~/my_fig.png'
            close: Bool, closes the figure when set.
        """
        plt.savefig(path, dpi=200)
        if close:
            plt.close(self.fig)

    def save_to_PIL(self, close=True):
        """
        saves the current figure toa PIL image object.
        Parameters:
            path: path to save at. Including extension. eg. '~/my_fig.png'
            close: Bool, closes the figure when set.
        """
        buf = io.BytesIO()
        self.save(buf, close=close)
        buf.seek(0)
        return Image.open(buf)

    def save_ax(self, row, col, path, close=True):
        """
        saves an axis-sub image.
        Parameters:
            row: the row to plot the image
            col: the clolumn to plot the image
            path: path to save at. Including extension. eg. '~/my_fig.png'
            close: Bool, closes the figure when set.
        """
        extent = (
            self.axs[row, col]
            .get_window_extent()
            .transformed(self.fig.dpi_scale_trans.inverted())
        )
        self.fig.savefig(path, bbox_inches=extent)
        if close:
            plt.close(self.fig)

    def save_ax_to_PIL(self, row, col, close=True):
        """
        saves an axis-sub image to a PIL image object.
        Parameters:
            row: the row to plot the image
            col: the clolumn to plot the image\
            close: Bool, closes the figure when set.
        """
        buf = io.BytesIO()
        self.save_ax(row, col, buf, close=close)
        buf.seek(0)
        img = Image.open(buf)

        # at this point the image still has a title. We search for the white background pixels, and crop the image smaller.
        img_array = np.array(img)
        H, W, _ = img_array.shape
        img_start = 0
        img_end = H - 1
        for h in range(0, H):
            if not (img_array[h, 0] == np.array([255, 255, 255, 255])).all():
                img_start = h
                break
        for h in range(H - 1, 0, -1):
            if not (img_array[h, 0] == np.array([255, 255, 255, 255])).all():
                img_end = h + 1
                break

        return img.crop((0, img_start, W, img_end))
