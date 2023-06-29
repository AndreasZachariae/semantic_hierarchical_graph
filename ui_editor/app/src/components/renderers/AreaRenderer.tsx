import { Area } from "../../types/Area";
import { Position } from "../../types/Position";

interface AreaRendererProps {
    areas: Area[];
    origin: Position;
    width: number;
    height: number;

    pixelWidth: number;
    pixelHeight: number;
};

export function AreaRenderer(props: AreaRendererProps) {

    return (
        <>
            <svg
                viewBox={`${-props.origin.x} ${-props.origin.y} ${props.width} ${props.height}`}
                xmlns={"http://www.w3.org/2000/svg"}
                style={{
                    position: "absolute",
                    width: `${props.pixelWidth}px`,
                    height: `${props.pixelHeight}px`,
                }}
            >

            </svg>
        </>
    )
}