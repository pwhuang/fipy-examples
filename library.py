from fipy.terms.abstractConvectionTerm import _AbstractConvectionTerm
from fipy.terms.abstractUpwindConvectionTerm import _AbstractUpwindConvectionTerm
from fipy.terms import TransientTermError

from fipy.variables.faceVariable import FaceVariable
from fipy.tools.dimensions.physicalField import PhysicalField
from fipy.tools import inline
from fipy.tools import numerix

class _DownwindConvectionTermAlpha(FaceVariable):
    def __init__(self, P):
        FaceVariable.__init__(self, mesh=P.mesh, elementshape=P.shape[:-1])
        self.P = self._requires(P)

    if inline.doInline:
        def _calcValue(self):
            P  = self.P.numericValue
            alpha = self._array.copy()

            inline._runInline("""
                alpha[i] = 0.5;

                if (P[i] < 0.) {
                    alpha[i] = 1.;
                } else {
                    alpha[i] = 0.;
                }
            """,
            alpha=alpha, P=P,
            ni = len(P.flat))

            return self._makeValue(value=alpha)
    else:
        def _calcValue(self):
            P  = self.P.numericValue
            alpha = numerix.where(P < 0., 1., 0.)
            return PhysicalField(value=alpha)

class _AbstractDownwindConvectionTerm(_AbstractConvectionTerm):
    def _alpha(self, P):
        return _DownwindConvectionTermAlpha(P)
    
class DownwindConvectionTerm(_AbstractDownwindConvectionTerm):
    r"""
    The discretization for this :class:`~fipy.terms.term.Term` is given by

    .. math::

       \int_V \nabla \cdot (\vec{u} \phi)\,dV \simeq \sum_{f} (\vec{n}
       \cdot \vec{u})_f \phi_f A_f

    where :math:`\phi_f=\alpha_f \phi_P +(1-\alpha_f)\phi_A` and
    :math:`\alpha_f` is calculated using the upwind convection scheme.
    For further details see :ref:`sec:NumericalSchemes`.
    """

    def _getDefaultSolver(self, var, solver, *args, **kwargs):
        solver = solver or super(DownwindConvectionTerm, self)._getDefaultSolver(var, solver, *args, **kwargs)
        if solver and not solver._canSolveAsymmetric():
            import warnings
            warnings.warn("%s cannot solve asymmetric matrices" % solver)
        from fipy.solvers import DefaultAsymmetricSolver
        return solver or DefaultAsymmetricSolver(*args, **kwargs)
    
class ExplicitDownwindConvectionTerm(_AbstractUpwindConvectionTerm):
    r"""
    The discretization for this :class:`~fipy.terms.term.Term` is given by

    .. math::

       \int_V \nabla \cdot (\vec{u} \phi)\,dV \simeq \sum_{f} (\vec{n}
       \cdot \vec{u})_f \phi_f A_f

    where :math:`\phi_f=\alpha_f \phi_P^\text{old} +(1-\alpha_f)\phi_A^\text{old}` and
    :math:`\alpha_f` is calculated using the upwind scheme.
    For further details see :ref:`sec:NumericalSchemes`.
    """

    def _getOldAdjacentValues(self, oldArray, id1, id2, dt):
        if dt is None:
            raise TransientTermError
        return numerix.take(oldArray, id2), numerix.take(oldArray, id1)

    def _getWeight(self, var, transientGeomCoeff=None, diffusionGeomCoeff=None):
        weight = _AbstractUpwindConvectionTerm._getWeight(self, var, transientGeomCoeff, diffusionGeomCoeff)
        if 'implicit' in list(weight.keys()):
            weight['explicit'] = weight['implicit']
            del weight['implicit']

        return weight