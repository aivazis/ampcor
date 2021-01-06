// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2021 all rights reserved


// externals
import React from 'react'


// create a context for maintaining the mosaic zoom level
const Context = React.createContext({
    // the reference to the mosaic
    mosaicRef: null,
    // the raster shape
    rasterShape: [0,0],
    // the shape of the tile grid
    gridShape: [0,0],
    // the shape of the mosaic at the current zoomlevel
    mosaicShape: [0,0],
    // the shape of the mosaic tiles
    tileShape: [0,0],

    // the current zoom level
    zoom: 0,
    // the minimum zoom level
    minZoom: 0,
    // the maximum zoom level
    maxZoom: Infinity,

    // if the client doesn't set up a context, setting the zoom level makes me
    setZoom() {
        // complain
        throw new Error("setZoom: no context provider")
    },

    // the current pan
    pan: null,

    // again, if there is no context provider
    setPan() {
        // complain
        throw new Error("setPan: no context provider")
    },
})


// the mosaic context provider factory
const Provider = ({
    // the reference to the mosaic
    mosaicRef,
    // the raster and tile shapes
    rasterShape, tileShape,
    // the zoom range and initial value
    minZoom=0, initZoom=Infinity, maxZoom=Infinity,
    // my children
    children
}) => {
    // build and access the current {zoom} level state
    const [zoom, setZoom] = React.useState(initZoom)
    // build and access the current {pan}
    const [pan, setPan] = React.useState(null)

    // convert the zoom level into a scaling factor for the raster hsape
    const scale = 1 << zoom

    // the shape of the tile grid is computed by rounding down the raster shape to
    // the nearest tile multiple
    // N.B.: this clips the tiles that ride on the right and bottom margin of the data set;
    // they aren't whole tiles, so they either need a dynamically generated margin or more
    // complicated request logic
    const gridShape = rasterShape.map((shp, i) => Math.floor(shp / tileShape[i] / scale))
    // from this, we can derive the shape of the mosaic
    const mosaicShape = gridShape.map((shp, i) => shp * tileShape[i])

    // set up the current value of the context
    const context = {
        // the reference to the mosaic
        mosaicRef,
        // the raster and tile shapes
        rasterShape, tileShape,
        // the shape of the mosaic and the grid of tiles
        mosaicShape, gridShape,
        // the zoom range and value
        minZoom, zoom, maxZoom,
        // the zoom level mutator
        setZoom,
        // the pan
        pan,
        // the pan mutator
        setPan,
    }

    // build the context and make it available to my {children}
    return (
        <Context.Provider value={context}>
            {children}
        </Context.Provider>
    )
}


// hook that provides access to the current mosaic shape
const useMosaic = () => {
    // pull the values from the context
    const { gridShape, mosaicShape } = React.useContext(Context)
    // make them available
    return { gridShape, mosaicShape }
}


// hook that provides access to the pan
const usePan = () => {
    // pull  the values from the context
    const { pan, setPan } = React.useContext(Context)
    // and make them available
    return [ pan, setPan ]
}


// hook that provides access to the current zoom level and its mutators
const useZoom = () => {
    // pull
    const {
        // the handle to the mosaic
        mosaicRef,
        // the shape of the mosaic,
        mosaicShape,
        // the zoom range and current value
        minZoom, zoom, maxZoom,
        // the zoom level mutator
        setZoom,
        // the pan mutator
        setPan,
    // from the context
    } = React.useContext(Context)

    // build the callbacks that tie the zoom controls to my state
    // zoom in is clipped at {minZoom}
    const zoomIn = () => {
        // zoom in is clipped at {minZoom}, so if we are arleady there
        if (zoom <= minZoom ) {
            // bail
            return
        }
        // adjust the zoom level
        setZoom(zoom-1)

        // get the {mosaic} viewport
        const view = mosaicRef.current
        // get its shape
        const width = view.clientWidth
        const height = view.clientHeight
        // and the scroll position of the upper left hand corner
        const scrollTop = view.scrollTop
        const scrollLeft = view.scrollLeft

        // unpack the shape of the displayed raster
        const [gheight, gwidth] = mosaicShape
        // compute the new scroll position
        const top = 2 * scrollTop + Math.min(height, gheight) - Math.min(height/2, gheight)
        const left = 2 * scrollLeft + Math.min(width, gwidth) - Math.min(width/2, gwidth)
        // record it
        setPan({ top, left })

        // all done
        return
    }

    const zoomOut = () => {
        // zoom out is clipped at {maxZoom}, so if we are already there
        if (zoom >= maxZoom) {
            // bail
            return
        }
        // adjust the level
        setZoom(zoom+1)

        // get the {mosaic}
        const view = mosaicRef.current
        // get its shape
        const width = view.clientWidth
        const height = view.clientHeight
        // and the scroll position of the upper left hand corner
        const scrollTop = view.scrollTop
        const scrollLeft = view.scrollLeft

        // unpack the shape of the displayed raster
        const [gheight, gwidth] = mosaicShape
        // target pixel
        const cx = scrollLeft + Math.min(width,gwidth)/2
        const cy = scrollTop + Math.min(height,gheight)/2

        // compute the new scroll position
        const top = scrollTop/2 + Math.min(height, gheight)/4 - Math.min(height, gheight/2)/2
        const left = scrollLeft/2 + Math.min(width, gwidth)/4 - Math.min(width, gwidth/2)/2
        // record it
        setPan({ top, left })

        // all done
        return
    }

    // and publish
    return { zoom, zoomIn, zoomOut }
}


// publish
export default { Context, Provider, useZoom, useMosaic, usePan }


// end of file
