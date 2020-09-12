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
    Select visualizations of intermediate ampcor products
    """


    # my workflow
    flow = ampcor.specs.ampcor()
    flow.doc = "the ampcor workflow"

    # other user configurable state
    # the size of the viewport
    viewport = ampcor.properties.tuple(schema=ampcor.properties.int())
    viewport.default = (800, 600)
    viewport.doc = "the size of the central plot window, in pixels"

    zoom = ampcor.properties.int(default=0)
    zoom.doc = "the zoom level"

    # arena rendering
    tileSize = ampcor.properties.tuple(schema=ampcor.properties.int())
    tileSize.default = 5,5
    tileSize.doc = "the render size of individual pixels"

    tileMargin = ampcor.properties.tuple(schema=ampcor.properties.int())
    tileMargin.default = 1,1
    tileMargin.doc = "the spacing between pixels"

    colorLow = ampcor.properties.array()
    colorLow.default = [0.2, 0.2, 0.2]
    colorLow.doc = "the color that corresponds to low values"

    colorHigh = ampcor.properties.array()
    colorHigh.default = [0.89, 0.52, 0.22] # pyre orange
    colorHigh.default = [1.0, 0.0, 0.0]
    colorHigh.doc = "the color that corresponds to high values"

    # colormap
    bins = ampcor.properties.int(default=5)
    bins.doc = "the number of value bins in the color map"

    # the raster grid spacing
    rulers = ampcor.properties.tuple(schema=ampcor.properties.int())
    rulers.default = 500, 500
    rulers.doc = "the raster grid spacing"


    # behaviors
    @ampcor.export(tip="static visualization of the coarse correlation surface")
    def coarse(self, plexus, **kwds):
        """
        Produce a visualization of the coarse correlation surface
        """
        # get the correlator
        correlator = self.flow.correlator
        # it has access to the secondary raster shape
        rasterShape = correlator.secondary.shape
        # and the workplan
        map, plan = correlator.plan()

        # the number of pairs
        pairs = len(plan.tiles)
        # and the padding
        pad = correlator.padding
        # determine the shape of the {gamma} arena
        shape = (pairs, 2*pad[0]+1, 2*pad[1]+1)
        # and its origin
        origin = (0, -pad[0], -pad[1])
        # print the values
        for spot in map[1]:
            # so the user can scroll
            print(spot)

        # make an empty arena
        gamma = ampcor.products.newArena()
        # configure it
        gamma.setSpec(origin=origin, shape=shape)
        gamma.data = "coarse_gamma.dat"
        # attach it to its data file
        gamma.open()

        # prime my content
        content = self.renderGamma(gamma=gamma, locations=map[1], rasterShape=rasterShape)
        # build the page
        page = self.page(title="the correlation surface", content=content)

        # open the output file
        with open(f"coarse_gamma.html", "w") as stream:
            # render the document and write it
            print('\n'.join(self.weaver.weave(document=page)), file=stream)

        # all done
        return 0


    @ampcor.export(tip="static visualization of the zoomed correlation surface")
    def zoomed(self, plexus, **kwds):
        """
        Produce a visualization of the zoomed correlation surface
        """
        # get the output
        offsets = self.flow.offsetMap
        # open its raster
        offsets.open()

        # get the correlator
        correlator = self.flow.correlator
        # it has access to the secondary raster shape
        rasterShape = correlator.secondary.shape
        # and the workplan
        map, plan = correlator.plan()

        # get the number of pairs
        pairs = len(plan.tiles)
        # the refine margin
        margin = correlator.refineMargin
        # the refine factor
        refine = correlator.refineFactor
        # and the zoom factor
        zoom = correlator.zoomFactor

        # the size of a zoomed gamma tile
        ext = (2*margin*refine + 1) * zoom
        # and its origin
        org = -margin * refine * zoom

        # build the shape of the zoomed gamma arena
        shape = (pairs, ext, ext)
        # and its origin
        origin = (0, org, org)

        gamma = ampcor.products.newArena()
        # configure it
        gamma.setSpec(origin=origin, shape=shape)
        gamma.data = "zoomed_gamma.dat"
        # attach it to its data file
        gamma.open()

        # we are going to place the zoomed correlation surface rendering on the output pixel
        locations = [
            [ int(ref+delta) for ref,delta in zip(offsets[idx].ref, offsets[idx].delta) ]
            for idx,_,_ in plan.tiles
            ]

        # prime my content
        content = self.renderGamma(gamma=gamma,
                                   locations=locations, rasterShape=rasterShape, zoom=refine*zoom)
        # build the page
        page = self.page(title="the zoomed correlation surface", content=content)

        # open the output file
        with open(f"zoomed_gamma.html", "w") as stream:
            # render the document and write it
            print('\n'.join(self.weaver.weave(document=page)), file=stream)

        # all done
        return 0


    @ampcor.export(tip="static visualization of the workplan")
    def plan(self, plexus, **kwds):
        """
        Visualize the workplan
        """
        # prime my content
        contnent = self.workplan()
        # build the page
        page = self.page(title="the workplan", content=content)

        # open the output file
        with open(f"plan.html", "w") as stream:
            # render the document and write it
            print('\n'.join(self.weaver.weave(document=page)), file=stream)

        # all done
        return 0


    @ampcor.export(tip="static visualization of the workplan")
    def offsets(self, plexus, **kwds):
        """
        Visualize the sequence of shift proposed by ampcor
        """
        # prime my content
        content = self.shifts()
        # build the page
        page = self.page(title="the shifts", content=content)

        # open the output file
        with open(f"offsets.html", "w") as stream:
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
    def shifts(self):
        """
        Visualize the shifts proposed by ampcor
        """
        yield ''
        # all done
        return


    def workplan(self):
        """
        Build a visualization of the workplan
        """
        # get the correlator
        correlator = self.flow.correlator
        # unpack the shapes
        chip = correlator.chip
        pad = correlator.padding
        # get the secondary raster
        secondary = correlator.secondary
        # unpack its shape
        height, width = secondary.shape

        zf = 2**self.zoom

        # wrappers
        yield '<!-- the plot frame -->'
        yield '<div class="iVu">'
        # add the plot element
        yield f'<!-- the plan -->'
        yield f'<div class="plot">'
        # with the drawing
        yield f'  <svg class="gamma" version="1.1"'
        yield f'       height="{height*zf}px"'
        yield f'       width="{width*zf}px"'
        yield f'       xmlns="http://www.w3.org/2000/svg">'
        yield f'    <g transform="scale({zf} {zf})">'

        gridSpacing = 500

        # make a horizontal grid
        hgrid = ' '.join([ f"M 0 {y} h {width}" for y in range(0, height, gridSpacing) ])
        # and render it
        yield f'    <path class="hgrid-major" d="{hgrid}" />'

        # make a verical grid
        vgrid = ' '.join([ f"M {x} 0 v {height}" for x in range(0, width, gridSpacing) ])
        # and render it
        yield f'    <path class="vgrid-major" d="{vgrid}" />'

        # print the coordinates of the grid intersections
        yield '<!-- grid intersections -->'
        for line in range(0, height, gridSpacing):
            for sample in range(0, width, gridSpacing):
                yield f'<text class="grid"'
                yield f'  y="{line}" x="{sample}" '
                yield f'  >'
                yield f'(line={line}, sample={sample})'
                yield f'</text>'

        # ask the correlator for the workplan
        map, plan = correlator.plan()

        # the shape of a reference tile
        reftileDY, reftileDX = chip
        sectileDY, sectileDX = [ c + 2*p for c,p in zip(chip, pad) ]
        # go through the map
        for (refLine, refSample), (secLine, secSample) in zip(*map):
            # plot the center
            yield f'<circle class="plan_ref"'
            yield f'  cx="{refSample}" cy="{refLine}" r="5"'
            yield f'  />'
            # shift to form the origin of the reference tile
            reftileY = refLine - reftileDY // 2
            reftileX = refSample - reftileDX // 2
            # generate the reference rectangle
            yield f'<rect class="plan_ref"'
            yield f'  x="{reftileX}" y="{reftileY}"'
            yield f'  width="{reftileDX}" height="{reftileDY}"'
            yield f'  rx="1" ry="1"'
            yield f'  >'
            yield f'  <title>({refLine},{refSample})+({reftileDY},{reftileDX})</title>'
            yield f'</rect>'

            # plot the center
            yield f'<circle class="plan_sec"'
            yield f'  cx="{secSample}" cy="{secLine}" r="5"'
            yield f'  />'
            # shift to form the origin of the secondary tile
            sectileY = secLine - sectileDY // 2
            sectileX = secSample - sectileDX // 2
            # generate the reference rectangle
            yield f'<rect class="plan_sec"'
            yield f'  x="{sectileX}" y="{sectileY}"'
            yield f'  width="{sectileDX}" height="{sectileDY}"'
            yield f'  rx="1" ry="1"'
            yield f'  >'
            yield f'  <title>({secLine},{secSample})+({sectileDY},{sectileDX})</title>'
            yield f'</rect>'

            # plot the shift
            yield f'<path class="plan_shift"'
            yield f'  d="M {refSample} {refLine} L {secSample} {secLine}"'
            yield f'  />'

        # close up the wrappers
        yield f'    </g>'
        yield '  </svg>'
        yield '</div>'
        yield '</div>'
        # all done
        return


    def renderGamma(self, gamma, locations, rasterShape, zoom=1):
        """
        Draw the given correlation surface
        """
        # unpack the raster shape
        height, width = rasterShape
        # get the number of pairs
        pairs = len(locations)

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
            colormaps[pid] = ColorMap(grid=grid, start=self.colorLow, end=self.colorHigh)

        # the outermost container
        yield f'<div class="iVu">'

        # sign on
        yield f'<!-- the correlation surface plot -->'
        # rendering a gamma
        yield f'<div class="plot">'
        yield f'  <svg class="gamma" version="1.1" xmlns="http://www.w3.org/2000/svg"'
        yield f'       height="{height}px"'
        yield f'       width="{width}px"'
        yield f'    >'
        # render a grid
        yield from self.renderGrid(shape=rasterShape)
        # render the arena
        yield from self.renderArena(arena=gamma,
                                    locations=locations, colormaps=colormaps, zoom=zoom)
        # done with the rendering of the correlation surface
        yield '  </svg>'
        yield '</div>'

        # make a grid
        grid = GeometricGrid(bins=self.bins)
        # and use it to make a color map for the legend
        colormap = ColorMap(grid=grid, start=self.colorLow, end=self.colorHigh)
        # make a legend
        legend = Legend(colormap=colormap)
        # ask for its bounding box
        lbox = legend.box

        # render he legend
        yield f'<!-- the legend for the correlation surface values -->'
        yield f'<svg class="gamma_legend" version="1.1" xmlns="http://www.w3.org/2000/svg"'
        yield f'     width="{lbox[0]}px" height="{lbox[1]}px"'
        yield f'    >'
        # render
        yield from legend.render()
        # close up the legend
        yield "</svg>"

        # close {iVu}
        yield '</div>'

        # all done
        return


    def renderArena(self, arena, locations, colormaps, zoom):
        """
        Generate a rendering of the data in an {arena} at the specified {locations}
        """
        # unpack the rendering characteristics
        tileSize = self.tileSize
        tileMargin = self.tileMargin
        # form the total space allocated to a tile
        tileBox = tuple(s + 2*m for s,m in zip(tileSize, tileMargin))

        # index the {arena}
        for idx in arena.layout:
            # unpack the index partially
            pid, *tdx = idx

            # build the coordinates that correspond to this tile by looking up the central spot
            center = locations[pid]
            # and shifting by the tile index
            pixel = tuple(c + i/zoom for c,i in zip(center, tdx))

            # we render pixels as tiles by zooming and shifting; the math below guarantees that
            # the center of the tile at index (0,0) sits on the pixel that corresponds to this
            # tile; form the coordinates of the upper left hand corner of the tile box
            boxOrigin = tuple(c - b//2 + i*b for c,b,i in zip(center, tileBox, tdx))
            # the tile origin is the box origin shifted by the margin
            tileOrigin = tuple(b + m for b,m in zip(boxOrigin, tileMargin))

            # get the data value
            value = arena[idx]
            # get the color that corresponds to this value
            color = colormaps[pid].color(value=value)

            # render
            yield f''
            # sign on
            yield f'<!-- arena: {tuple(idx)} <- {value}  @{color} -->'
            # make a colored tile
            yield f'<rect y="{tileOrigin[0]}" x="{tileOrigin[1]}"'
            yield f'      height="{tileSize[0]}" width="{tileSize[1]}" rx="1" ry="1"'
            yield f'      fill="{color}" stroke="none"'
            yield f'  >'
            # decorate it so the user can get tile details by hovering over it
            yield f'  <title>'
            yield f'pixel: {pixel}'
            yield f'index: {tuple(idx)}'
            yield f'value: {value}'
            yield f'  </title>'
            # done with tile
            yield f'</rect>'

        # all done
        return


    def renderGrid(self, shape):
        """
        Draw a grid on the raster display
        """
        # unpack the shape
        height, width = shape
        # and the rulers
        vSpacing, hSpacing = self.rulers

        # if the user hasn't disable the horizontal grid
        if vSpacing is not None and vSpacing > 0:
            # make a horizontal grid
            hgrid = ' '.join([ f"M 0 {y} h {width}" for y in range(0, height, vSpacing) ])
            # and render it
            yield f'    <path class="hgrid-major" d="{hgrid}" />'

        # similarly for the vertical grid
        if hSpacing is not None and hSpacing > 0:
            # make it
            vgrid = ' '.join([ f"M {x} 0 v {height}" for x in range(0, width, hSpacing) ])
            # and render it
            yield f'    <path class="vgrid-major" d="{vgrid}" />'

        # print the coordinates of the grid intersections
        yield '<!-- grid intersections -->'
        for line in range(0, height, vSpacing):
            for sample in range(0, width, hSpacing):
                yield f'<text class="grid"'
                yield f'  y="{line}" x="{sample}" '
                yield f'  >'
                yield f'(line={line}, sample={sample})'
                yield f'</text>'

        # all done
        return


    # implementation details
    def page(self, title, content):
        """
        The page builder
        """
        # start the document
        yield from self.start()
        # my body
        yield from self.body(title=title, content=content)
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


    def body(self, title, content):
        """
        The document body
        """
        # open the tag
        yield "  <body>"
        # the page header
        yield from self.header(title=title)
        # draw the frame
        yield from content
        # the page footer
        yield from self.footer()
        # close up
        yield "  </body>"

        # all done
        return


    def header(self, title):
        """
        Generate the app header
        """
        yield '<header>'
        yield f'  <!-- app id -->'
        yield f'  <p>'
        yield f'    {title}'
        yield f'  </p>'
        yield f'</header>'
        yield f''

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
        yield '<!-- logo -->'
        yield '<div>'
        yield '<svg id="logo" version="1.1" xmlns="http://www.w3.org/2000/svg">'
        yield '<g '
        yield '    fill="#f37f19" fill-opacity="0.5"'
        yield '    stroke="none"'
        yield '    transform="scale(0.15 0.15)">'
        yield '<path d="'
        yield 'M 100 0'
        yield 'C 200 75 -125 160 60 300'
        yield 'C 20 210 90 170 95 170'
        yield 'C 80 260 130 225 135 285'
        yield 'C 160 260 160 250 155 240'
        yield 'C 180 260 175 270 170 300'
        yield 'C 205 270 240 210 195 135'
        yield 'C 195 165 190 180 160 200'
        yield 'C 175 180 220 55 100 0'
        yield 'Z'
        yield '" />'
        yield '</g>'
        yield '</svg>'
        yield '</div>'
        yield '</footer>'
        yield ''

        # all done
        return


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
        # exactly that
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


    # metamethods
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


class Legend:
    """
    An object that can render a legend
    """


    # public state
    # sizes in pixels
    λ = 20, 15        # atomic entry size
    pad = 5,5         # my padding
    ticks = 30, 1     # my tick marks
    valueWidth = 4    # the width of a legend label
    fontSize = 10     # the size of the font used to render the tick values

    @property
    def box(self):
        """
        Compute my bounding box
        """
        # get my grid
        g = self.colormap.grid
        # my stretch size is
        height = (
            # my top pad
            self.pad[1] +
            # my entry height multiplied by its replication factor
            self.λ[1] * sum(g.powers) +
            # the space taken by the tickmarks
            3 * self.ticks[1] * len(g.ticks) +
            # my bottom pad
            self.pad[1]
            )
        # my fixed size
        width = (
            # my left pad
            self.pad[0] +
            # my tick marks are longer than the tiles :)
            self.ticks[0] - self.ticks[1] +
            # another margin
            self.pad[1] +
            # the width of my value formatting
            self.valueWidth * self.fontSize +
            # my right pad
            self.pad[0]
            )

        # all done
        return width, height


    # interface
    def render(self):
        """
        Render me
        """
        # get mey geometry
        λ = self.λ
        pad = self.pad
        ticks = self.ticks

        # get my color map
        map = self.colormap
        # and its grid
        g = map.grid
        # ask the grid for the number of bins
        bins = g.bins

        # compute my size
        width, height = self.box

        # initialize the cursor; this is the position of the first tick mark
        cursor = [pad[0], height-pad[1]]

        # go through the bins
        for bin in range(bins):
            # make the tick mark
            yield from self.tickmark(cursor=cursor)
            # render the tick value
            yield from self.tickvalue(cursor=cursor, value=f"{g.ticks[bin]:4.2f}")
            # move the cursor to the UL corner of the til
            cursor[1] -= 3*self.ticks[1] + λ[1] * g.powers[bin]

            # compute the height of this tile
            height = λ[1] * g.powers[bin]
            # get its color
            color = map.rgb(bin=bin)
            # render the tile
            yield from self.tile(cursor=cursor, height=height, color=color)

        # make the last tick mark
        yield from self.tickmark(cursor=cursor)
        # render the tick value
        yield from self.tickvalue(cursor=cursor, value=f"{g.ticks[bins]:4.2f}")

        # all done
        return


    # metamethods
    def __init__(self, colormap, λ=λ, pad=pad, **kwds):
        # chain up
        super().__init__(**kwds)

        # save the colormap
        self.colormap = colormap
        # and my geometry
        self.λ = λ
        self.pad = pad

        # all done
        return


    # implementation details
    def tile(self, cursor, height, color):
        """
        Render a bin
        """
        # make a rounded rectangle
        yield f'<rect class="legend_tile"'
        yield f'    x="{cursor[0]}" y="{cursor[1]}"'
        yield f'    width="{self.λ[0]}" height="{height}"'
        yield f'    rx="1" ry="1"'
        yield f'    fill="{color}"'
        yield f'  />'

        # all done
        return


    def tickmark(self, cursor):
        """
        Render a tick mark
        """
        # make a line
        yield f'<path class="legend_tick"'
        yield f'  d="M {cursor[0]-self.ticks[1]} {cursor[1]-self.ticks[1]} h {self.ticks[0]}"'
        yield f'  />'
        # all done
        return


    def tickvalue(self, cursor, value):
        """
        Render the value associated with a tick
        """
        # open the text tag
        yield f'<text class="legend_value"'
        yield f'  x="{cursor[0]+self.ticks[0]+self.pad[1]}" y="{cursor[1]+2}"'
        yield f'  >'
        # place the value
        yield value
        # close the tag
        yield f'</text>'
        # all done
        return


# end of file
