// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import path from 'path'
import React, { useRef, useState } from 'react'
// locals
import styles from './style'
import { Tile } from 'widgets'


// currently, a tile is an {img} that's lazy loaded from a {uri}
const widget = ({uri, origins, zoomin, style}) => {
    // content styling
    // mix the viewport style
    const viewportStyle = {
        // using the supplied and local styles
        ...styles.viewport, ...style?.viewport,
    }
    // mix the mosaic style
    const mosaicStyle = {
        // using the suppplied and local styles
        ...styles.mosaic, ...style?.mosaic,
    }
    // finally, the tile style is a mix of
    const tileStyle = {
        // the supplied styling
        ...styles.tile, ...style?.tile,
    }

    // fold the indices in a tile origin into a string
    const foldRanks = (first, ...rest) => rest.reduce(
        // by looping over the indices and tacking on each one to the partial result
        (partial, index) => `${partial}x${index}`,
        // which is primed by converting the first index into a string
        first.toString()
        )
    // given the origin of a tile, build its uri
    const tileURI = (origin) => path.join(uri, foldRanks(...origin))

    // make a reference to attach to my container
    const viewportRef = useRef(null)
    // on double click, center to the mouse coordinates
    const center = ({clientX, clientY}) => {
        // get the viewport
        const viewport = viewportRef?.current
        // if it has been rendered
        if (viewport) {
            // get the viewport boundinbox
            const box = viewport.getBoundingClientRect()

            // compute the location of the click relative to the viewport
            const x = clientX - box.left
            const y = clientY - box.top
            // get the size of the viewport
            const width = viewport.clientWidth
            const height = viewport.clientHeight
            // get the current scroll position
            const left = viewport.scrollLeft
            const top = viewport.scrollTop
            // if i have a valid zoom callback
            if (zoomin) {
                // invoke it
                // zoomin()
            }
            // and scroll to the new location
            viewport.scroll({
                top: top + y - height/2,
                left: left + x - width/2,
                behavior: "smooth",
            })
        }
    }

    // render
    return (
        <div ref={viewportRef} style={viewportStyle} onDoubleClick={center} >
            <div style={mosaicStyle} >
                {origins.map(origin => {
                     // form the tile uri
                     const uri = tileURI(origin)
                     // render
                     return (
                         <Tile key={uri} uri={uri} style={tileStyle} />
                     )
                 })}
            </div>
        </div>
    )
}


// publish
export default widget


// end of file