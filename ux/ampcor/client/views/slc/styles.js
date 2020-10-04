// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// get colors
import { wheel, theme } from '~/palette'
// get the base styles
import base from '~/views/styles'


// tool styling; see '~/widgets/tool/*.js' for details
const tool = {
    // the base state
    base: {
        box: {
            flex: "none",
            width: "2em",
            height: "2em",
            color: theme.banner.name,
            // backgroundColor: theme.page.background,
        },

        group: {
            transform: "scale(0.4)",
        },

        frame: {
            fill: wheel.gray.obsidian,
            fillOpacity: 0.4,
            stroke: wheel.gray.steel,
            strokeOpacity: 0.75,
            strokeWidth: "5px",
        },

        icon: {
            stroke: theme.banner.name,
            strokeOpacity: 0.75,
            strokeWidth: "5px",
        },

    },
    inactive: {
        frame: {
            fillOpacity: .2,
        },
        icon: {
            strokeOpacity: .2,
        },
    },
    selected: {
        frame: {
            fillOpacity: 1.0,
            stroke: theme.banner.name,
        },
    }
}


// publish
export default {
    // the container
    slc: {
        // inherit
        ...base.panel,
    },

    viewport: {
        // inherit
        ...base.viewport,
    },

    plot: {
        flex: "none",
        display: "flex",
        flexDirection: "row",
        flexWrap: "wrap",
        overflow: "hidden",
    },

    tile: {
        flex: "none",
        // the sizes here are defaults; they get overriden when more/better is known
        width: "256px",
        height: "256px",
    },

    // slc tool box styling
    slcToolbox: {
        toolbox: {
            // positioning
            position: "absolute",
            right: "1.0em",
            top: "0.75em",
            // size
            width: "2.0em",
            height: "6.3em", // 3 tools at 2em each, plus some space in between
            // styling
            opacity: "0.75",
            // backgroundColor: theme.page.background,
            // border: `1px solid ${wheel.gray.basalt}`,
        },

        // tool styling; see '~/widgets/tool/*.js' for details
        tool: {
            // start with the default established up top
            ...tool
        },
    },

    // slc tool box styling
    zoomToolbox: {
        toolbox: {
            // positioning
            position: "absolute",
            right: "1.0em",
            bottom: "0.75em",
            // size
            width: "2.0em",
            height: "4.0em", // 2 tools at 2em each, no space in between
            // styling
            opacity: "0.75",
            // backgroundColor: theme.page.background,
            // border: `1px solid ${wheel.gray.basalt}`,
        },

        // tool styling; see '~/widgets/tool/*.js' for details
        tool: {
            // start with the default established up top
            ...tool
        },
    },

    placeholder: {
        // inherit
        ...base.placeholder,
    },

}


// end of file
