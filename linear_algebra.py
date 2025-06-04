import numpy as np
import numpy.typing as npt


Z_ROTATION_180 = np.array(
    [
        [-1, 0, 0],
        [0, -1, 0],
        [0,  0, 1]
    ],
    dtype=float
)

Z_UNIT_VECTOR = np.array([0.0, 0.0, 1.0])

Z_REFLECTOR = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0,-1],
    ],
    dtype=float
)

IDENTITY_MATRIX = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    dtype=float
)


class LinearAlgebra:


    @staticmethod
    def normalize(vector: npt.NDArray) -> npt.NDArray:
        """Return the corresponding unit vector for a given vector.

        Notes
        -----
        The normalization of a vector in Euclidean geometry to its 
        corresponding unit vector is accomplished by dividing the 
        vector by its Euclidean norm, given as the square root of the sum
        of its components squared.

        The normalization of a matrix to a unit matrix is accomplished by
        dividing the matrix by its Frobenius norm, a generalization of the
        Euclidean norm to an N by M matrix. It can be calculated by taking the
        trace of the conjugate transpose of the matrix times the matrix itself.
        """

        norm = np.linalg.norm(vector)

        if norm != 0:
            return vector / norm
        else:
            return vector


    @staticmethod
    def quaternion_builder(
        vector: npt.ArrayLike,
        angle: float,
    ) -> npt.NDArray:
        """Build a quaternion matrix.

        Parameters
        ----------
        vector : ArrayLike
            The vector to rotate around (can be list or NDArray)
        angle : float
            The angle to rotate by (in radians)

        Returns
        -------
        quaternion_matrix : NDArray
            A quaternion matrix that rotates by `angle` around `vector`
        """

        quaternion = [
            np.cos(angle / 2),
            vector[0] * np.sin(angle / 2),
            vector[1] * np.sin(angle / 2),
            vector[2] * np.sin(angle / 2),
        ]

        # The normalized quaternion vector
        norm_quat = LinearAlgebra.normalize(quaternion)

        quaternion_matrix = np.array(
            [
                [
                    1 - 2 * (norm_quat[2] ** 2 + norm_quat[3] ** 2),
                    2 * (norm_quat[1] * norm_quat[2] - norm_quat[0] * norm_quat[3]),
                    2 * (norm_quat[1] * norm_quat[3] + norm_quat[0] * norm_quat[2]),
                ],
                [
                    2 * (norm_quat[1] * norm_quat[2] + norm_quat[0] * norm_quat[3]),
                    1 - 2 * (norm_quat[1] ** 2 + norm_quat[3] ** 2),
                    2 * (norm_quat[2] * norm_quat[3] - norm_quat[0] * norm_quat[1]),
                ],
                [
                    2 * (norm_quat[1] * norm_quat[3] - norm_quat[0] * norm_quat[2]),
                    2 * (norm_quat[2] * norm_quat[3] + norm_quat[0] * norm_quat[1]),
                    1 - 2 * (norm_quat[1] ** 2 + norm_quat[2] ** 2),
                ],
            ]
        )

        return quaternion_matrix


    @staticmethod
    def vector_angle(vector_one: npt.ArrayLike, vector_two: npt.ArrayLike) -> float:
        """Calculate the angle (in radians) between two vectors."""

        unit_vector_one = LinearAlgebra.normalize(vector_one)
        unit_vector_two = LinearAlgebra.normalize(vector_two)

        angle = np.arccos(
            np.clip(
                np.dot(unit_vector_one, unit_vector_two),
                -1.0,
                1.0,
            )
        )

        return float(angle)


    @staticmethod
    def distance_calculator(
        vector_one: npt.ArrayLike, vector_two: npt.ArrayLike
    ) -> np.float64:
        """Calculate the Euclidean distance between two vectors."""

        distance = np.absolute(
            np.sqrt(
                ((float(vector_one[0]) - float(vector_two[0])) ** 2)
                + ((float(vector_one[1]) - float(vector_two[1])) ** 2)
                + ((float(vector_one[2]) - float(vector_two[2])) ** 2)
            )
        )

        return distance


    @staticmethod
    def nudge_matrix_generator(stage: str) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Creates the nudge matrices and vector for avoiding local minima during optimization.

        Parameters
        ----------
        stage : {"Reactant", "Product", "Transition"}
            Stage of the reaction.

        Returns
        -------
        nudge_rotate_x : NDArray
            An array that rotates around the x-axis by +/- 15 degrees
        nudge_rotate_y : NDArray
            An array that rotates around the y-axis by +/- 20 degrees
        nudge_translate : NDArray
            A vector that nudges the molecule by a set amount

        Notes
        -----
        Calling this for `stage="Reactant"` yields rotation matrices that rotate by +15 degrees
        in the X-axis and +20 degrees in the Y-axis. The opposite is true for `stage="Product"`.
        Calling this for `stage="Transition"` returns the identity matrix for rotations. 
        """

        # xv_degree (15 Degrees)
        xv_degree = np.pi / 12
        # xx_degree (20 Degrees)
        xx_degree = np.pi / 9

        nudge_translate = np.array([0, 0, 0])

        if stage == "Reactant":
            nudge_rotate_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(xv_degree), -np.sin(xv_degree)],
                    [0, np.sin(xv_degree), np.cos(xv_degree)],
                ],
                dtype=float,
            )

            nudge_rotate_y = np.array(
                [
                    [np.cos(xx_degree), 0, np.sin(xx_degree)],
                    [0, 1, 0],
                    [-np.sin(xx_degree), 0, np.cos(xx_degree)],
                ],
                dtype=float,
            )
        elif stage == "Product":
            nudge_rotate_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(-xv_degree), -np.sin(-xv_degree)],
                    [0, np.sin(-xv_degree), np.cos(-xv_degree)],
                ],
                dtype=float,
            )

            nudge_rotate_y = np.array(
                [
                    [np.cos(-xx_degree), 0, np.sin(-xx_degree)],
                    [0, 1, 0],
                    [-np.sin(-xx_degree), 0, np.cos(-xx_degree)],
                ],
                dtype=float,
            )
        elif stage == "Transition":
            nudge_rotate_x = IDENTITY_MATRIX
            nudge_rotate_y = IDENTITY_MATRIX

        return nudge_rotate_x, nudge_rotate_y, nudge_translate


    @staticmethod
    def gen_alignment_matrix(
        vector_one: npt.ArrayLike,
        vector_two: npt.ArrayLike | None = None,
        alignment_angle: float | None = None,
    ):
        """Generate a matrix to align two vectors.

        Parameters
        ----------
        vector_one : ArrayLike
            The vector that will be aligned.
        vector_two : ArrayLike | None, default=None
            The vector to which `vector_one` will be aligned. Default aligns to the Z-axis.

        Notes
        -----
        This method calls `LinearAlgebra.vector_angle(vector_one, vector_two)`,
        then `LinearAlgebra.normalize(np.cross(vector_one, vector_two))`, and feeds the resulting
        alignment vector and alignment angle to `LinearAlgebra.quaternion_builder()`.
        """

        if vector_two is None:
            vector_two = Z_UNIT_VECTOR

        if alignment_angle is None:
            alignment_angle = LinearAlgebra.vector_angle(vector_one, vector_two)

        alignment_vector = LinearAlgebra.normalize(np.cross(vector_one, vector_two))

        alignment_quaternion = LinearAlgebra.quaternion_builder(alignment_vector, alignment_angle)

        return alignment_quaternion