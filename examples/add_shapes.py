"""Adding shapes to the explorer."""
import matplotlib.pyplot as plt
import covariogram_explorer as ce

kite = ce.ConvexPolygon([(0, -0.5), (0.35, 0), (0, 0.6), (-0.35, 0)], "Kite", "kite")
rhombus = ce.Parallelogram((0.7, 0), (0.35, 0.35 * 3**0.5), "Rhombus", "rhombus",
                           "a parallelogram")

app = ce.App(ce.make_shapes() + [kite, rhombus])
app.show_gallery()
plt.show()
