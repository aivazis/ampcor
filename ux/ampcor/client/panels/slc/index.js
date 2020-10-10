// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import path from 'path'
import React, { useState } from 'react'
// locals
import styles from './styles'
import { Mosaic, SLC, Zoom } from '~/widgets'


// a panel that displays an SLC along with it controls
const panel = ({uri, shape, style}) => {
    // the shape of my tiles; optimized so that each line takes up one page
    const tileShape = [512, 512]

    // set up the maximum zoom level
    const maxZoom = Math.max(
        // divide each rank by the tile size, figure out how many times we can double the
        // base tile along that rank, and make sure the smaller dimension dominates
        Math.min(
            ...shape.map((shp, idx) => Math.floor(Math.log2(shp/tileShape[idx])))
        ),
        // but clip it to zero, since very small raasters should be shown at full zoom
        0
    )

    // the zoom control will manage the zoom level; we need state and callback
    // start out at maximum zoom
    const [zoom, setZoom] = useState(maxZoom)
    // convert the zoom level into a scaling factor for the raster hsape
    const scale = 1 << zoom
    // build the callbacks that tie the zoom controls to my state
    // zoom in is clipped at level 0
    const zoomin = () => (zoom > 0 ? setZoom(zoom-1) : null)
    // zoom out is clipped at {maxZoom}
    const zoomout = () => (zoom < maxZoom ? setZoom(zoom+1) : null)

    // the signal control selects the tile content; pick "amplitude" as the default
    const [signal, setSignal] = useState("amplitude")

    // the shape of the tile grid is computed by rounding down the raster shape to
    // the nearest tile multiple
    // N.B.: this clips the tiles that ride on the right and bottom margin of the data set;
    // they aren't whole tiles, so they either need a dynamically generated margin or more
    // complicated request logic
    const gridShape = shape.map((shp, i) => Math.floor(shp / tileShape[i] / scale))
    // from this, we can derive the shape of the mosaic
    const mosaicShape = gridShape.map((shp, i) => shp * tileShape[i])

    // rendering the tiles involving assembling the correct tile uri for each one
    // each tile spec has an index in the tile grid starting at (0,0) with shape {gridShape}
    // each rank in this index is multiplied by the {tileShape} to deduce the location of the
    // corresponding raster tile

    // part of the problem requires folding an array with shape or origin information into a string
    const foldRanks = (first, ...rest) => rest.reduce(
        // each operation takes the accumulated value and tacks on the next rank
        (sofar, rank) => `${sofar}x${rank}`,
        // when primed with the first rank
        `${first}`
    )

    // part of the problem involves computing the exterior product of two arreys
    // e.g:
    //
    //   [0, 1, 2] x [0, 1] -> [ [0,0], [0,1], [1,0], [1,1], [2, 0], [2, 1] ]
    //
    // which lets us assemble the set of indices of all the tiles in the grid, if we
    // know how many we want along each direction
    // here is how you do this kind of thing for N arrays
    const product = (first, ...rest) => rest.reduce(
        // each operation loops over all arrays except the first one, and uses
        //   - {prefix}: an array with whatever has been accumulated so far; this is g
        //   - {arg}:    the current array that is being processed
        //   - {argIdx}: and the index of this array in {rest}, which one less than its index
        //               in the original argument list (since we set {first} aside)
        (prefix, arg, argIndex) => prefix.flatMap(
            // loop over the contents of the array with the partial answer, turning each {entry}
            // into an array formed by looping over the current {arg}
            prefixEntry => arg.map(
                // adding each {arg} entry next to each {prefix} entry and flattening the result
                argEntry => [prefixEntry, argEntry].flat()
            )
        ),
        // starting with: prefix = range(len(first))
        first
    )

    // now, {gridShape} has the dimensions of the tile grid; let's turn it into an array
    // of indices along each rank, and while we are there, multiply by the {tileShape}
    // to get all the tile origins
    const indices = gridShape.map(
        // by taking each rank and using it to allocate an array of that size and fill it
        // with consecuitve integers, i.e. map {shp} to {range(shp)}
        (shp, rank) => Array(shp).fill().map((_,i) => i * tileShape[rank])
    )

    // indices now has two arrays; each one is the set of origin coordinates along each
    // rank; so compute their {product} to get the origins of all the tiles
    const origins = product(...indices)

    // we now know enough to assemble the uri seed
    const seedURI = path.join(uri, signal, zoom.toString(), foldRanks(...tileShape))

    // style mixing
    // mix the panel style
    const panelStyle = {
        ...styles.panel, ...style?.panel,
    }
    // content styling
    // mix the canvas style
    const viewportStyle = {
        // using the supplied and local styles
        ...styles.viewport, ...style?.viewport,
    }
    // mix the mosaic style
    const mosaicStyle = {
        // using the suppplied and local styles
        ...styles.mosaic, ...style?.mosaic,
        // and the computed shape
        ...{ height: mosaicShape[0], width: mosaicShape[1] },
    }
    // finally, the tile style is a mix of
    const tileStyle = {
        // the supplied styling
        ...styles.tile, ...style?.tile,
        // and the computed shape
        ...{ height: tileShape[0], width: tileShape[1] },
    }
    // assemble the styles in a single object
    const contentsStyle = {
        viewport: viewportStyle,
        mosaic: mosaicStyle,
        tile: tileStyle,
    }

    // render
    return (
        <div style={panelStyle}>
            {/* the SLC signal toolbox */}
            <SLC select={setSignal} style={styles.slcToolbox} />
            {/* the zoom toolbox */}
            <Zoom zoomin={zoomin} zoomout={zoomout} style={styles.zoomToolbox} />
            {/* the raster display */}
            <Mosaic uri={seedURI} origins={origins} style={contentsStyle} zoomin={zoomin} />
        </div>
    )
}


// publish
export default panel


// end of file