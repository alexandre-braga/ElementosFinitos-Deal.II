#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dg_vector.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_trace.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>

#include <fstream>
#include <iostream>
#include <cmath>

const double viscosity = 1.0;

namespace AulasDealII
{
    using namespace dealii;

    template <int dim>
    class SourceTerm : public Function<dim>
    {
    public:
        SourceTerm() : Function<dim>(dim + 1) {}

        double visc = 1.0;
        virtual void vector_value(const Point<dim> &p,
                                  Vector<double> &values) const
        {
            for (unsigned int i = 0; i < dim; ++i)
            {
                values[i] = dim * M_PI * M_PI * (visc - 1.0);
                if (i == dim - 1)
                {
                    values[i] = -dim * M_PI * M_PI * ((dim - 1.0) * visc + 1.0);
                }

                for (unsigned int j = 0; j < dim; ++j)
                {
                    if (i == j)
                    {
                        values[i] *= sin(M_PI * p[j]);
                    }
                    else
                    {
                        values[i] *= cos(M_PI * p[j]);
                    }
                }
            }
            values[dim] = 0.0;
        }
    };

    template <int dim>
    class VelocityBoundaryValues : public Function<dim>
    {
    public:
        VelocityBoundaryValues() : Function<dim>(dim) {}

        virtual void vector_value(const Point<dim> &p,
                                  Vector<double> &values) const
        {
            for (unsigned int i = 0; i < dim; ++i)
            {
                values[i] = 1.0;
                if (i == dim - 1)
                {
                    values[i] = -(dim - 1);
                }
                for (unsigned int j = 0; j < dim; ++j)
                {
                    if (i == j)
                    {
                        values[i] *= sin(M_PI * p[j]);
                    }
                    else
                    {
                        values[i] *= cos(M_PI * p[j]);
                    }
                }
            }
        }
    };

    template <int dim>
    class ExactSolution : public Function<dim>
    {
    public:
        ExactSolution() : Function<dim>(dim + 1) {}
        virtual void vector_value(const Point<dim> &p,
                                  Vector<double> &values) const
        {
            Assert(values.size() == dim + 1,
                   ExcDimensionMismatch(values.size(), dim + 1));

            for (unsigned int i = 0; i < dim; ++i)
            {
                values[i] = 1.0;
                if (i == dim - 1)
                {
                    values[i] = -(dim - 1);
                }

                for (unsigned int j = 0; j < dim; ++j)
                {
                    if (i == j)
                    {
                        values[i] *= sin(M_PI * p[j]);
                    }
                    else
                    {
                        values[i] *= cos(M_PI * p[j]);
                    }
                }
            }

            values[dim] = dim * M_PI;
            for (int i = 0; i < dim; ++i)
            {
                values[dim] *= cos(M_PI * p[i]);
            }
        }
    };

    template <int dim>
    class MixedLaplaceProblem
    {
    public:
        MixedLaplaceProblem(const unsigned int degree)
            : degree(degree),
              fe_local(FE_DGQ<dim>(degree), dim,
                       FE_DGQ<dim>(degree), 1),
              dof_handler_local(triangulation),
              fe(FE_TraceQ<dim>(degree), dim,
                 FE_DGQ<dim>(0), 1),
              dof_handler(triangulation)
        {
        }

        void run(int i, ConvergenceTable &convergence_table)
        {
            make_grid_and_dofs(i);
            assemble_system(true);
            solve();
            assemble_system(false);
            compute_errors(convergence_table);
            output_results();
        }

    private:
        const unsigned int degree;

        Triangulation<dim> triangulation;

        FESystem<dim> fe_local;
        DoFHandler<dim> dof_handler_local;
        Vector<double> solution_local;

        FESystem<dim> fe;
        DoFHandler<dim> dof_handler;
        Vector<double> solution;
        Vector<double> system_rhs;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;

        AffineConstraints<double> constraints;

        void make_grid_and_dofs(int i)
        {
            GridGenerator::hyper_cube(triangulation, 0, 1.0);
            triangulation.refine_global(i);

            dof_handler.distribute_dofs(fe);
            dof_handler_local.distribute_dofs(fe_local);

            std::cout << "Number of active cells: "
                      << triangulation.n_active_cells()
                      << std::endl
                      << "Total number of cells: "
                      << triangulation.n_cells()
                      << std::endl
                      << "Number of degrees of freedom for the multiplier: "
                      << dof_handler.n_dofs()
                      << std::endl
                      << std::endl;

            solution.reinit(dof_handler.n_dofs());
            system_rhs.reinit(dof_handler.n_dofs());
            solution_local.reinit(dof_handler_local.n_dofs());

            DynamicSparsityPattern dsp(dof_handler.n_dofs());
            DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
            sparsity_pattern.copy_from(dsp);
            system_matrix.reinit(sparsity_pattern);

            constraints.clear();
            DoFTools::make_hanging_node_constraints(dof_handler, constraints);
            ExactSolution<dim> solution_function;
            FEValuesExtractors::Vector velocity(0);
            VectorTools::interpolate_boundary_values(dof_handler, 0, solution_function, constraints, fe.component_mask(velocity));
            constraints.close();
            //       constraints.print(std::cout);
        }

        void assemble_system(bool globalProblem)
        {
            QGauss<dim> quadrature_formula(degree + 2);
            QGauss<dim - 1> face_quadrature_formula(degree + 2);

            FEValues<dim> fe_values_local(fe_local, quadrature_formula,
                                          update_values | update_gradients |
                                              update_quadrature_points | update_JxW_values);
            FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                             update_values | update_normal_vectors |
                                                 update_quadrature_points | update_JxW_values);
            FEFaceValues<dim> fe_face_values_local(fe_local, face_quadrature_formula,
                                                   update_values | update_gradients | update_normal_vectors |
                                                       update_quadrature_points | update_JxW_values);

            const unsigned int dofs_local_per_cell = fe_local.dofs_per_cell;
            const unsigned int n_q_points = quadrature_formula.size();
            const unsigned int n_face_q_points = face_quadrature_formula.size();

            std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);

            FullMatrix<double> A_matrix(dofs_local_per_cell, dofs_local_per_cell);
            FullMatrix<double> B_matrix(dofs_local_per_cell, fe.dofs_per_cell);
            FullMatrix<double> BT_matrix(fe.dofs_per_cell, dofs_local_per_cell);
            FullMatrix<double> C_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
            Vector<double> F_vector(dofs_local_per_cell);
            Vector<double> U_vector(dofs_local_per_cell);
            Vector<double> cell_vector(fe.dofs_per_cell);

            FullMatrix<double> aux_matrix(dofs_local_per_cell, fe.dofs_per_cell);
            Vector<double> aux_vector(dofs_local_per_cell);

            std::vector<Tensor<1, dim>> mult_values_u(face_quadrature_formula.size());
            std::vector<double> mult_values_p(face_quadrature_formula.size());

            Vector<double> intp(dofs_local_per_cell);

            const SourceTerm<dim> source_term;
            const VelocityBoundaryValues<dim> velocity_boundary_values;

            std::vector<Vector<double>> st_values(n_q_points, Vector<double>(dim + 1));
            std::vector<Vector<double>> boundary_values(face_quadrature_formula.size());

            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(dim);

            for (const auto &cell : dof_handler.active_cell_iterators())
            {

                const double h_mesh = cell->diameter();
                // std::cout << "h: " << h_mesh << std::endl;
                const double beta0 = 35.;
                const double beta_stab_u = viscosity * std::pow(2, degree - 1) * beta0 / h_mesh;
                const double beta_stab_p = 1.;

                typename DoFHandler<dim>::active_cell_iterator loc_cell(&triangulation, cell->level(), cell->index(), &dof_handler_local);

                fe_values_local.reinit(loc_cell);

                A_matrix = 0;
                F_vector = 0;
                B_matrix = 0;
                BT_matrix = 0;
                C_matrix = 0;
                U_vector = 0;

                source_term.vector_value_list(fe_values_local.get_quadrature_points(), st_values);

                intp.reinit(dofs_local_per_cell);

                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    for (unsigned int i = 0; i < dofs_local_per_cell; ++i)
                    {
                        const Tensor<2, dim> grad_phi_i_u = fe_values_local[velocities].gradient(i, q);
                        const double div_phi_i_u = fe_values_local[velocities].divergence(i, q);
                        const double phi_i_p = fe_values_local[pressure].value(i, q);

                        intp(i) += phi_i_p * fe_values_local.JxW(q);

                        for (unsigned int j = 0; j < dofs_local_per_cell; ++j)
                        {
                            const Tensor<2, dim> grad_phi_j_u = fe_values_local[velocities].gradient(j, q);
                            const double div_phi_j_u = fe_values_local[velocities].divergence(j, q);
                            const double phi_j_p = fe_values_local[pressure].value(j, q);

                            A_matrix(i, j) += viscosity * scalar_product(grad_phi_i_u, grad_phi_j_u) * fe_values_local.JxW(q); // (grad u, grad v)

                            A_matrix(i, j) -= div_phi_i_u * phi_j_p * fe_values_local.JxW(q); // -(p, div v)

                            A_matrix(i, j) -= phi_i_p * div_phi_j_u * fe_values_local.JxW(q); // -(q, div u)

                            //      A_matrix(i,j) -= 0.000000001 * phi_i_p * phi_j_p * fe_values_local.JxW(q);            // -eps*(p, q)
                        }

                        const unsigned int component_i =
                            fe_local.system_to_component_index(i).first;

                        F_vector(i) += fe_values_local.shape_value(i, q) * st_values[q](component_i) * fe_values_local.JxW(q); // (f, v)
                    }
                }

                // comento depois
                for (unsigned int i = 0; i < dofs_local_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_local_per_cell; ++j)
                    {
                        A_matrix(i, j) += beta_stab_p * intp(i) * intp(j);
                    }
                }

                /// Loop nas faces dos elementos
                for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell; ++face_n)
                {
                    fe_face_values.reinit(cell, face_n);
                    fe_face_values_local.reinit(loc_cell, face_n);
                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {

                        const double JxW = fe_face_values_local.JxW(q);
                        const Tensor<1, dim> normal = fe_face_values.normal_vector(q);

                        for (unsigned int i = 0; i < dofs_local_per_cell; ++i)
                        {
                            const Tensor<2, dim> grad_phi_i_u = fe_face_values_local[velocities].gradient(i, q);
                            const Tensor<1, dim> phi_i_u = fe_face_values_local[velocities].value(i, q);
                            const double phi_i_p = fe_face_values_local[pressure].value(i, q);

                            for (unsigned int j = 0; j < dofs_local_per_cell; ++j)
                            {
                                const Tensor<2, dim> grad_phi_j_u = fe_face_values_local[velocities].gradient(j, q);
                                const Tensor<1, dim> phi_j_u = fe_face_values_local[velocities].value(j, q);
                                const double phi_j_p = fe_face_values_local[pressure].value(j, q);

                                A_matrix(i, j) += (-viscosity * operator*(grad_phi_i_u, normal) * phi_j_u  // - (grad v n, u)
                                                   - phi_i_u * viscosity * operator*(grad_phi_j_u, normal) // - (grad u n, v)
                                                   + phi_j_p * phi_i_u * normal                            // (p, v.n)
                                                   + phi_i_p * phi_j_u * normal                            // (q, u.n)
                                                   + beta_stab_u * phi_i_u * phi_j_u                       // beta* (u, v)
                                                   ) *
                                                  JxW;
                            }
                        }
                    }
                }
                if (globalProblem)
                {
                    // comento depois
                    for (unsigned int i = 0; i < dofs_local_per_cell; ++i)
                    {
                        B_matrix(i, fe.dofs_per_cell - 1) -= beta_stab_p * intp(i);
                    }

                    C_matrix(fe.dofs_per_cell - 1, fe.dofs_per_cell - 1) -= beta_stab_p;

                    for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell; ++face_n)
                    {
                        fe_face_values.reinit(cell, face_n);
                        fe_face_values_local.reinit(loc_cell, face_n);

                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                        {
                            const double JxW = fe_face_values.JxW(q);
                            const Tensor<1, dim> normal = fe_face_values.normal_vector(q);

                            for (unsigned int i = 0; i < dofs_local_per_cell; ++i)
                            {
                                const Tensor<2, dim> grad_phi_i_u = fe_face_values_local[velocities].gradient(i, q);
                                const Tensor<1, dim> phi_i_u = fe_face_values_local[velocities].value(i, q);
                                const double phi_i_p = fe_face_values_local[pressure].value(i, q);

                                for (unsigned int j = 0; j < fe.dofs_per_cell - 1; ++j)
                                {
                                    const Tensor<1, dim> phi_j_m = fe_face_values[velocities].value(j, q);

                                    B_matrix(i, j) += viscosity * (phi_j_m * operator*(grad_phi_i_u, normal) // (lamb, grad v n)
                                                                   - phi_i_p * phi_j_m * normal              // - (lamb, q n)
                                                                   - beta_stab_u * phi_j_m * phi_i_u         // - beta *(lamb, v)
                                                                   ) *
                                                      JxW;
                                }
                            }
                            for (unsigned int i = 0; i < fe.dofs_per_cell - 1; ++i)
                            {
                                const Tensor<1, dim> phi_i_m = fe_face_values[velocities].value(i, q);

                                for (unsigned int j = 0; j < fe.dofs_per_cell - 1; ++j)
                                {
                                    const Tensor<1, dim> phi_j_m = fe_face_values[velocities].value(j, q);

                                    /// Negativo para facilitar a condensacao estatica
                                    C_matrix(i, j) -= (beta_stab_u * phi_i_m * phi_j_m //  beta_u *(lamb_u, mu_u)
                                                       ) *
                                                      JxW;
                                }
                            }
                        }
                    }
                }
                else
                {
                    fe_face_values[pressure].get_function_values(solution, mult_values_p);
                    const double multp = mult_values_p[0];
                    //   std::cout << "mult: " << multp << std::endl;

                    for (unsigned int i = 0; i < dofs_local_per_cell; ++i)
                    {
                        F_vector(i) += beta_stab_p * intp(i) * multp;
                    }
                    for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell; ++face_n)
                    {
                        fe_face_values.reinit(cell, face_n);
                        fe_face_values_local.reinit(loc_cell, face_n);
                        fe_face_values[velocities].get_function_values(solution, mult_values_u);

                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                        {
                            const double JxW = fe_face_values.JxW(q);
                            const Tensor<1, dim> normal = fe_face_values.normal_vector(q);

                            for (unsigned int i = 0; i < dofs_local_per_cell; ++i)
                            {
                                const Tensor<1, dim> phi_i_u = fe_face_values_local[velocities].value(i, q);
                                const Tensor<2, dim> grad_phi_i_u = fe_face_values_local[velocities].gradient(i, q);
                                const double phi_i_p = fe_face_values_local[pressure].value(i, q);

                                F_vector(i) += (-viscosity * operator*(grad_phi_i_u, normal) * mult_values_u[q] + beta_stab_u * phi_i_u * mult_values_u[q] + phi_i_p * normal * mult_values_u[q]) * JxW;
                            }
                        }
                    }
                }

                A_matrix.gauss_jordan();

                if (globalProblem)
                {
                    // problema simétrico
                    BT_matrix.copy_transposed(B_matrix);
                    A_matrix.vmult(aux_vector, F_vector, false);     //  A^{-1} * F
                    BT_matrix.vmult(cell_vector, aux_vector, false); //  B.T * A^{-1} * F
                    A_matrix.mmult(aux_matrix, B_matrix, false);     //  A^{-1} * B
                    BT_matrix.mmult(C_matrix, aux_matrix, true);     // -C + B.T * A^{-1} * B

                    //   C_matrix.print(std::cout,8);
                    //   std::cout << " " <<  std::endl;

                    cell->get_dof_indices(dof_indices);

                    constraints.distribute_local_to_global(C_matrix,
                                                           cell_vector,
                                                           dof_indices,
                                                           system_matrix, system_rhs);
                }
                else
                {
                    A_matrix.vmult(U_vector, F_vector, false);

                    loc_cell->set_dof_values(U_vector, solution_local);
                }
            }
        }

        void solve()
        {

            PreconditionJacobi<SparseMatrix<double>> precondition;
            precondition.initialize(system_matrix, 0.6);

            SolverControl solver_control(system_matrix.m() * 10, 1e-10 * system_rhs.l2_norm());
            SolverCG<> solver(solver_control);
            solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

            std::cout << std::endl
                      << "   Number of BiCGStab iterations: " << solver_control.last_step()
                      << std::endl
                      << std::endl;

            constraints.distribute(solution);
        }

        void compute_errors(ConvergenceTable &convergence_table)
        {
            const ComponentSelectFunction<dim>
                pressure_mask(dim, dim + 1);
            const ComponentSelectFunction<dim>
                velocity_mask(std::make_pair(0, dim), dim + 1);

            ExactSolution<dim> exact_solution;
            Vector<double> cellwise_errors(triangulation.n_active_cells());

            QGauss<dim> quadrature(degree + 2);

            double u_l2_error, p_l2_error;

            VectorTools::integrate_difference(dof_handler_local, solution_local, exact_solution,
                                              cellwise_errors, quadrature,
                                              VectorTools::L2_norm,
                                              &velocity_mask);

            u_l2_error = VectorTools::compute_global_error(triangulation,
                                                           cellwise_errors,
                                                           VectorTools::L2_norm);

            VectorTools::integrate_difference(dof_handler_local, solution_local, exact_solution,
                                              cellwise_errors, quadrature,
                                              VectorTools::L2_norm,
                                              &pressure_mask);

            p_l2_error = VectorTools::compute_global_error(triangulation,
                                                           cellwise_errors,
                                                           VectorTools::L2_norm);

            std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
                      << ",   ||e_u||_L2 = " << u_l2_error
                      << std::endl
                      << std::endl;

            convergence_table.add_value("cells", triangulation.n_active_cells());
            convergence_table.add_value("L2_u", u_l2_error);
            convergence_table.add_value("L2_p", p_l2_error);
        }

        void output_results() const
        {
            std::string filename = "Results/solution_local.vtu";
            std::ofstream output(filename.c_str());

            DataOut<dim> data_out;
            std::vector<std::string> names(dim, "Velocity");
            names.emplace_back("Pressure");
            std::vector<DataComponentInterpretation::DataComponentInterpretation> component_interpretation(dim + 1, DataComponentInterpretation::component_is_part_of_vector);
            component_interpretation[dim] = DataComponentInterpretation::component_is_scalar;
            data_out.add_data_vector(dof_handler_local, solution_local, names, component_interpretation);

            data_out.build_patches(fe_local.degree);
            data_out.write_vtu(output);

            //        std::string   filename_face = "multiplier.vtu";
            //        std::ofstream face_output (filename_face.c_str());
            //
            //        DataOutFaces<dim> data_out_face(false);
            //        std::vector<std::string> face_name(dim,"Trace_velocity");
            //        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            //        face_component_type(dim, DataComponentInterpretation::component_is_part_of_vector);
            //
            //        data_out_face.add_data_vector (dof_handler,
            //                                       solution,
            //                                       face_name,
            //                                       face_component_type);
            //
            //        data_out_face.build_patches (fe.degree);
            //        data_out_face.write_vtu (face_output);
        }
    };

}

int main()
{
    using namespace dealii;
    using namespace AulasDealII;

    // dimensao
    const int dim = 2;

    ConvergenceTable convergence_table;

    for (int i = 2; i < 6; ++i)
    {
        // grau do polinomio
        MixedLaplaceProblem<dim> mixed_laplace_problem(1);
        mixed_laplace_problem.run(i, convergence_table);
    }

    convergence_table.set_precision("L2_u", 3);
    convergence_table.set_scientific("L2_u", true);
    convergence_table.evaluate_convergence_rates("L2_u", "cells", ConvergenceTable::reduction_rate_log2, dim);

    convergence_table.set_precision("L2_p", 3);
    convergence_table.set_scientific("L2_p", true);
    convergence_table.evaluate_convergence_rates("L2_p", "cells", ConvergenceTable::reduction_rate_log2, dim);

    convergence_table.write_text(std::cout);

    std::ofstream data_output("Results/taxas.dat");
    convergence_table.write_text(data_output);

    std::ofstream tex_output("Results/taxas.tex");
    convergence_table.write_tex(tex_output);

    return 0;
}
