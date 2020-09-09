# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# externals
import ampcor
import itertools
import journal


# clunky visualization of the correlation surface
class Gamma(ampcor.shells.command, family="ampcor.cli.gamma"):
    """
    Visualize the correlation surface
    """


    # my workflow
    flow = ampcor.specs.ampcor()
    flow.doc = "the ampcor workflow"

    # other user configurable state
    # the size of the viewport
    viewport = ampcor.properties.tuple(schema=ampcor.properties.int())
    viewport.default = (600, 800)

    # colormap
    bins = ampcor.properties.int(default=5)



    # behaviors
    @ampcor.export(tip="generate an html page with a static visualization of the coarse surface")
    def coarse(self, plexus, **kwds):
        """
        Produce an offset map between the {reference} and {secondary} images
        """
        # build the page
        page = self.page(content=self.gammaCoarse)

        # open the output file
        with open(f"coarse.html", "w") as stream:
            # render the document and write it
            print('\n'.join(self.weaver.weave(document=page)), file=stream)

        # all done
        return 0


    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)

        # instantiate a weaver
        weaver = ampcor.weaver.weaver(name=f"gamma.weaver")
        # configure it
        weaver.language = "html"
        # and attach it
        self.weaver = weaver

        # all done
        return


    # the types of content i can generate
    def gammaCoarse(self):
        """
        Build a visualization of the coarse correlation surface
        """
        # wrapper
        yield '<!-- the plot frame -->'
        yield '<div class="iVu">'

        # the drawing
        yield from self.gammaCoarseDraw()
        # draw the legend
        yield from self.legend(min=.0, max=1.00)

        # close up the wrapper
        yield "    </div>"

        # all done
        return


    def gammaCoarseDraw(self):
        """
        """
        # get the correlator
        correlator = self.flow.correlator
        # unpack the shapes
        chip = correlator.chip
        pad = correlator.padding
        # get the secondary raster
        secondary = correlator.secondary
        # get and open the offsets
        offsets = correlator.offsets.open()

        height, width = secondary.shape
        # add the plot element
        yield f'<!-- the map -->'
        yield f'<div class="plot"'
        yield f'    style="height: {self.viewport[0]}px; width: {self.viewport[1]}px">'
        yield f'  <svg class="gamma" version="1.1"'
        yield f'       height="{height}px"'
        yield f'       width="{width}px"'
        yield f'       xmlns="http://www.w3.org/2000/svg">'

        # ask the correlator for the workplan
        map, plan = correlator.plan()
        # the number of pairs
        pairs = len(plan.tiles)
        # hence the shape of the {gamma} arena is
        gammaShape = (pairs, 2*pad[0]+1, 2*pad[1]+1)
        # with origin at
        gammaOrigin = (0, -pad[0], -pad[1])

        for spot in map[1]:
            print(spot)

        # make a default instance
        gamma = ampcor.products.newArena()
        # configure it
        gamma.setSpec(origin=gammaOrigin, shape=gammaShape)
        gamma.data = "coarse_gamma.dat"
        # attach it to its file
        gamma.open()

        # to find the max correlation values
        highs = [ 0 ] * pairs
        # go through the product
        for idx in gamma.layout:
            # get each value
            g = gamma[idx]
            # get the pair id
            pid = idx[0]
            # if the value is the new high
            if g > highs[pid]:
                # replace it
                highs[pid] = g
        # to avoid spurious overflows, go through the highs
        for idx in range(len(highs)):
            # scale the highs up at the least significant figure
            highs[idx] *= 1 + 1e-6

        # build color maps
        colormaps = [ None ] * pairs
        # by going through all the pairs
        for pid in range(pairs):
            # to make a grid
            grid = GeometricGrid(max=highs[pid], bins=self.bins)
            # and a color map
            colormaps[pid] = ColorMap(grid=grid)

        # make a horizontal grid
        hgrid = ' '.join([ f"M 0 {y} h {width}" for y in range(0, height, 500) ])
        # and render it
        yield f'    <path class="hgrid-major" d="{hgrid}" />'

        # make a verical grid
        vgrid = ' '.join([ f"M {x} 0 v {height}" for x in range(0, width, 500) ])
        # and render it
        yield f'    <path class="vgrid-major" d="{vgrid}" />'

        # print the coordinates of the grid intersections
        yield '<!-- grid intersections -->'
        for line in range(0, height, 500):
            for sample in range(0, width, 500):
                yield f'<text class="grid"'
                yield f'  y="{line}" x="{sample}" '
                yield f'  >'
                yield f'(line={line}, sample={sample})'
                yield f'</text>'

        # now, go through all the entries in the product
        for pid, dx, dy in gamma.layout:
            # get the value of {gamma}
            cor = gamma[pid, dx, dy]
            # my color
            color = colormaps[pid].color(cor)
            # get the secondary pixel that corresponds to this {pid}
            spot = map[1][pid]
            # shift to get my center
            line = spot[0] + 7*dx
            sample = spot[1] + 7*dy
            yield f'    <rect class="gamma_value"'
            yield f'      x="{sample}" y="{line}"'
            yield f'      width="{5}" height="{5}" rx="1" ry="1"'
            yield f'      fill="{color}"'
            yield f'      >'
            yield f'      <title class="gamma_detail">'
            yield f'pos: ({spot[0] + dx}, {spot[1] + dy})'
            yield f'shift: ({dx}, {dy})'
            yield f'gamma: {cor:.2g}'
            yield f'      </title>'
            yield f'    </rect>'

        # close up the wrapper
        yield '  </svg>'
        yield '</div>'
        # all done
        return


    # implementation details
    def page(self, content):
        """
        The page builder
        """
        # start the document
        yield from self.start()
        # my body
        yield from self.body(content=content)
        # finish up
        yield from self.end()
        # all done
        return


    def start(self):
        """
        The beginning of the page
        """
        # generate
        yield '  <head>'
        yield '    <meta charset="utf-8"/>'
        yield '    <title>gamma</title>'
        yield '    <link href="styles/page.css" type="text/css" rel="stylesheet" media="all" />'
        yield '    <link href="styles/header.css" type="text/css" rel="stylesheet" media="all" />'
        yield '    <link href="styles/footer.css" type="text/css" rel="stylesheet" media="all" />'
        yield '    <link href="styles/gamma.css" type="text/css" rel="stylesheet" media="all" />'
        yield '  </head>'
        # all done
        return


    def end(self):
        """
        The bottom of the page
        """
        # nothing for now
        yield ""
        # all done
        return


    def body(self, content):
        """
        The document body
        """
        # open the tag
        yield "  <body>"
        # the page header
        yield from self.header()
        # draw the frame
        yield from content()
        # the page footer
        yield from self.footer()
        # close up
        yield "  </body>"

        # all done
        return


    def header(self):
        """
        Generate the app header
        """
        yield '<header>'
        yield '  <!-- app id -->'
        yield '  <p>'
        yield '    the correlation surface'
        yield '  </p>'
        yield '  <!-- logo -->'
        yield '  <img id="logo" src="logo.png"/>'
        yield '</header>'
        yield ''

        # all done
        return


    def footer(self):
        """
        The app footer
        """
        yield '<!-- colophon -->'
        yield '<footer>'
        yield '  <span class="copyright">'
        yield '    copyright &copy; 1998-2020'
        yield '    <a href="http://www.orthologue.com/michael.aivazis">michael aïvázis</a>'
        yield '    -- all rights reserved'
        yield '  </span>'
        yield '  <span class="social">'
        yield '  </span>'
        yield '</footer>'
        yield ''

        # all done
        return


    def legend(self, min, max):
        """
        Generate the legend
        """
        yield f'<!-- the legend -->'
        yield f'<svg class="gamma_legend" version="1.1"'
        yield  '     width="70px" height="350px"'
        yield  '     xmlns="http://www.w3.org/2000/svg">'

        # the length scale in pixels
        λ = 10
        # set the margin
        margin = 5
        # get the bins
        bins = self.bins
        # the cursor
        cursor = [margin, margin]

        # make a grid
        g = GeometricGrid(min=min, max=max, bins=bins, scale=2)
        # and a color map that rides on this grid
        c = ColorMap(g)

        # draw the top tick mark
        yield from self.tickmark(cursor)
        # and its value
        yield from self.tickvalue(cursor, value="max")

        values = ["0.00"] + [""]*(bins-1) + ["max"]

        bin = bins - 1
        # now loop to get the rest
        while bin >= 0:
            # get the scaling factor for this bin
            scale = g.powers[bin]
            # compute the height of the tile
            height = λ * scale
            # get the tick mark value
            tick = g.ticks[bin]
            # compute the color
            color = c.rgb(bin)
            # make a tile
            yield from self.legendTile(cursor=cursor, height=height, color=color)
            # draw the tick mark
            yield from self.tickmark(cursor)
            # and its value
            yield from self.tickvalue(cursor, value=values[bin])
            # update the counter
            bin -= 1

        # close up the wrapper
        yield "      </svg>"
        # all done
        return


    def legendTile(self, cursor, height, color):
        """
        Make a legend tile
        """
        # make a rounded rectangle
        yield f'<rect class="legend_tile"'
        yield f'    x="{cursor[0]}" y="{cursor[1]}"'
        yield f'    width="20" height="{height}"'
        yield f'    rx="1" ry="1"'
        yield f'    fill="{color}"'
        yield f'  />'

        # move the cursor
        cursor[1] += height

        # all done
        return


    def tickmark(self, cursor):
        # move the cursor
        cursor[1] += 1

        # make a line
        yield f'<path class="legend_tick"'
        yield f'  d="M {cursor[0]-1} {cursor[1]} h 30"'
        yield f'  />'

        # move the cursor
        cursor[1] += 1

        # all done
        return


    def tickvalue(self, cursor, value):
        # start the tag
        yield f'<text class="legend_value"'
        yield f'  x="{cursor[0]+35}" y="{cursor[1]+1}"'
        yield f'  >'
        # place the value
        yield value
        # close the tag
        yield f'</text>'


class GeometricGrid:
    """
    Underflow + N bins + overflow
    """

    def bin(self, value):
        """
        Figure out in which bin {value} falls

        Returns -1 for values less than {min}, and {bins+1} for values greater than {max}
        """
        # initialize the bin cursor
        cursor = -1
        # run up the tick marks; for small {bins}, it's faster than bisection
        for tick in self.ticks:
            # if the value is up to {tick}
            if value <= tick:
                # we found the bin
                break
            # otherwise, up the cursor and grab the next bin
            cursor += 1
        # all done
        if cursor >= self.bins:
            print(f"overflow: {value}, max={self.ticks[-1]}")
        return cursor


    def __init__(self, max=1, min=0, scale=2, bins=5, invert=False, **kwds):
        # chain up
        super().__init__(**kwds)

        # save the configuration parameters
        self.min = min
        self.max = max
        self.bins = bins
        self.scale = scale

        # the geometrical features of the grid that do not depend on the interval of choice
        # make a sequence of exponents
        exponents = range(bins) if invert is True else reversed(range(bins))
        # form a sequence of the powers of {scale} up to {bins}
        self.powers = tuple( scale**n for n in exponents )
        # the bin scale
        self.scale = 1 / sum(self.powers)
        # the widths of the bins
        self.weights = tuple( n * self.scale for n in self.powers )

        # map to the interval of choice
        interval = max - min
        # my resolution, i.e. the width of the smallest bin
        self.resolution = interval * self.scale
        # the widths of the bins
        self.intervals = tuple( interval*δ for δ in self.weights )
        # the tick marks
        self.ticks = tuple(itertools.accumulate(self.intervals, initial=self.min))

        # all done
        return


class ColorMap:
    """
    """

    # public data
    start = [0.20, 0.20, 0.20]
    end =   [1.00, 0.00, 0.00]
    end =   [0.68, 0.70, 0.32] # pyre green
    end =   [0.29, 0.67, 0.91] # pyre blue
    end =   [0.89, 0.52, 0.22] # pyre orange
    end =   [1.00, 0.00, 0.00] # red

    under = [0.00, 0.00, 0.00]
    over = [1.00, 1.00, 1.00]


    def color(self, value):
        """
        Ask the grid for the bin, then lookup the color
        """
        # exactly that...
        return self.rgb(self.grid.bin(value=value))


    def rgb(self, bin):
        """
        Convert a color tuple from my table into an html color spec
        """
        # on underflow
        if bin < 0:
            # use the corresponding color
            r,g,b = self.under
        # on overflow
        elif bin >= len(self.colors):
            # use the corresponding color
            r,g,b = self.over
        # otherwise
        else:
            # get the color values
            r,g,b = self.colors[bin]
        # format and return
        return f'rgb({int(100*r)}%, {int(100*g)}%, {int(100*b)}%)'


    def __init__(self, grid, start=start, end=end, **kwds):
        # chain up
        super().__init__(**kwds)

        # save the grid
        self.grid = grid

        # get the grid weights
        weights = grid.weights
        # make a sequence of parameter values
        ticks = [0] + list(reversed(weights[-len(weights)+1:-1])) + [1]
        # assign a color to each bin by biasing a gradient with the width of the grid bins
        self.colors = [ [ s + (e-s)*t for s,e in zip(start, end)] for t in ticks]

        # all done
        return


# end of file
