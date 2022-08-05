#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

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



namespace AulasDealII
{
using namespace dealii;


    template <int dim>
    class SourceTerm : public Function<dim>
    {
    public:
        SourceTerm () : Function<dim>(1) {}
        
        virtual double value (const Point<dim>   &p,
                              const unsigned int  /*component = 0*/) const
        {
            double return_value = dim * M_PI*M_PI;
            
            for (int i = 0; i < dim; ++i) {
                return_value *= sin(M_PI * p[i]);
            }
            
            return return_value;
        }
    };
    
    
    template <int dim>
    class BoundaryPressure : public Function<dim>
    {
    public:
        BoundaryPressure () : Function<dim>(1) {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  /*component = 0*/) const
        {
            double return_value = 1.0;
            for (int i = 0; i < dim; ++i)
            {
                return_value *= sin(M_PI * p[i]);
            }
            return return_value;
        }
    };

template <int dim>
class ExactSolution : public Function<dim>
{
public:
    ExactSolution () : Function<dim>(dim+1) {}
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const
    {
        Assert (values.size() == dim+1,
                ExcDimensionMismatch (values.size(), dim+1));
        
        for (unsigned int i = 0; i < dim; ++i)
        {
            values[i] = -M_PI;
            for (unsigned int j = 0; j < dim; ++j)
            {
                if (i == j)
                {
                    values[i] *= cos(M_PI * p[j]);
                }
                else
                {
                    values[i] *= sin(M_PI * p[j]);
                }
            }
        }
        
        values[dim] = 1.0;
        for (int i = 0; i < dim; ++i)
        {
            values[dim] *= sin(M_PI * p[i]);
        }
    }
};



template <int dim>
class MixedLaplaceProblem
{
public:
    MixedLaplaceProblem (const unsigned int degree)
        :
        degree (degree),
        fe_local (FE_DGQ<dim>(degree), dim,
                  FE_DGQ<dim>(degree), 1),
        dof_handler_local (triangulation),
        fe (degree),
        dof_handler (triangulation)
    {}

    void run (int i, ConvergenceTable &convergence_table)
    {
        make_grid_and_dofs(i);
        assemble_system (true);
        solve ();
        assemble_system (false);
        compute_errors (convergence_table);
        output_results ();
    }


private:

    const unsigned int   degree;

    Triangulation<dim>   triangulation;

    FESystem<dim>        fe_local;
    DoFHandler<dim>      dof_handler_local;
    Vector<double>       solution_local;

    FE_FaceQ<dim>        fe;
    DoFHandler<dim>      dof_handler;
    Vector<double>       solution;
    Vector<double>       system_rhs;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    AffineConstraints<double> constraints;

    
    void make_grid_and_dofs (int i)
    {
        GridGenerator::hyper_cube (triangulation, 0.0, 1.0);
        triangulation.refine_global (i);

        dof_handler.distribute_dofs (fe);
        dof_handler_local.distribute_dofs(fe_local);

        
        std::cout << "Number of active cells: "
                  << triangulation.n_active_cells()
                  << std::endl
                  << "Total number of cells: "
                  << triangulation.n_cells()
                  << std::endl
                  << "Number of degrees of freedom for the multiplier: "
                  << dof_handler.n_dofs()
                  << std::endl<< std::endl;


        solution.reinit (dof_handler.n_dofs());
        system_rhs.reinit (dof_handler.n_dofs());
        solution_local.reinit (dof_handler_local.n_dofs());

        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);

        constraints.clear ();
//        DoFTools::make_hanging_node_constraints (dof_handler, constraints);
        BoundaryPressure <dim> solution_function;
        VectorTools::interpolate_boundary_values (dof_handler, 0, solution_function, constraints);
        constraints.close ();
    }


    void assemble_system (bool globalProblem)
    {
        QGauss<dim>   quadrature_formula(degree+2);
        QGauss<dim-1> face_quadrature_formula(degree+2);

        FEValues<dim> fe_values_local (fe_local, quadrature_formula,
                                 update_values    | update_gradients |
                                 update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values_local (fe_local, face_quadrature_formula,
                                                update_values    | update_normal_vectors |
                                                update_quadrature_points  | update_JxW_values);

        const unsigned int   dofs_local_per_cell   = fe_local.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();

        std::vector<types::global_dof_index> dof_indices (fe.dofs_per_cell);

        FullMatrix<double>   cell_matrix (fe.dofs_per_cell, fe.dofs_per_cell);
        FullMatrix<double>   A_matrix (dofs_local_per_cell, dofs_local_per_cell);
        FullMatrix<double>   B_matrix (dofs_local_per_cell, fe.dofs_per_cell);
        FullMatrix<double>   BT_matrix (fe.dofs_per_cell, dofs_local_per_cell);
        Vector<double>       F_vector (dofs_local_per_cell);
        Vector<double>       U_vector (dofs_local_per_cell);
        Vector<double>       cell_vector (fe.dofs_per_cell);

        FullMatrix<double>   aux_matrix (dofs_local_per_cell, fe.dofs_per_cell);
        Vector<double>       aux_vector (dofs_local_per_cell);

        std::vector<double>  mult_values(face_quadrature_formula.size());


        const SourceTerm<dim>           source_term;

        std::vector<double> st_values (n_q_points);
        Tensor<2, dim> K;
        Tensor<2, dim> A;

        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);

        const double beta_stab_0 = -1.0;
        double beta_stab_p;
        const double del1 = -0.5;
        const double del2 =  0.5;
        const double del3 =  0.5;

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            const double h_mesh = cell->diameter()*std::sqrt(dim)/dim;

            typename DoFHandler<dim>::active_cell_iterator loc_cell (&triangulation, cell->level(), cell->index(), &dof_handler_local);
            fe_values_local.reinit (loc_cell);

            
            //Hydraulic Permeability
            K.clear();
            for (unsigned int d = 0; d < dim; ++d)
                K[d][d] = 1.;
            
            //Hydraulic Conductivity
            A = invert(K);

            source_term.value_list (fe_values_local.get_quadrature_points(),
                                    st_values);

            beta_stab_p =  beta_stab_0*linfty_norm(K)/h_mesh;

            A_matrix    = 0;
            F_vector    = 0;
            B_matrix    = 0;
            BT_matrix   = 0;
            U_vector    = 0;
            cell_matrix = 0;
            cell_vector = 0;


            for (unsigned int q=0; q<n_q_points; ++q)
            {

                for (unsigned int i=0; i<dofs_local_per_cell; ++i)
                {
                    const Tensor<1,dim> phi_i_u        = fe_values_local[velocities].value (i, q);
                    const auto rot_phi_i_u  = fe_values_local[velocities].curl (i, q);
                    const double        div_phi_i_u    = fe_values_local[velocities].divergence (i, q);
                    const double        phi_i_p        = fe_values_local[pressure].value (i, q);
                    const Tensor<1,dim> grad_phi_i_p   = fe_values_local[pressure].gradient(i, q);

                    for (unsigned int j=0; j<dofs_local_per_cell; ++j)
                    {
                        const Tensor<1,dim> phi_j_u      = fe_values_local[velocities].value (j, q);
                        const auto rot_phi_j_u  = fe_values_local[velocities].curl (j, q);
                        const double        div_phi_j_u  = fe_values_local[velocities].divergence (j, q);
                        const double        phi_j_p      = fe_values_local[pressure].value (j, q);
                        const Tensor<1,dim> grad_phi_j_p = fe_values_local[pressure].gradient(j, q);

                        A_matrix(i,j) += phi_i_u * A * phi_j_u * fe_values_local.JxW(q);     // ( u, v)
                        
                        A_matrix(i,j) -= div_phi_i_u * phi_j_p * fe_values_local.JxW(q);       // -(p, div v)
                        
                        A_matrix(i,j) -= phi_i_p * div_phi_j_u * fe_values_local.JxW(q);       // -(q, div u)
                        
                        A_matrix(i,j) += del1 * K * (A * phi_j_u + grad_phi_j_p) *
                                                (A * phi_i_u + grad_phi_i_p) *
                                                fe_values_local.JxW(q);  // -0.5 * K (gradp + Au) * (gradq + Av)

                        A_matrix(i,j) += del2 * linfty_norm(A) * div_phi_j_u *
                                                div_phi_i_u *
                                                fe_values_local.JxW(q);  // 0.5 * (div u, div v)
                        
                         A_matrix(i,j) += del3 * linfty_norm(A) * rot_phi_i_u * rot_phi_j_u *
                                                fe_values_local.JxW(q);  // 0.5 * (||K|| rot Au, rot Av)

                    }
                    F_vector(i) -= phi_i_p * st_values[q] * fe_values_local.JxW(q);    // -(f, q)
                    
                    F_vector(i) += del2 * linfty_norm(A) * div_phi_i_u *
                                    st_values[q] *
                                    fe_values_local.JxW(q);        // 0.5 * (f, div v)

                }
            }

            /// Loop nas faces dos elementos
            for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
            {
                fe_face_values.reinit (cell, face_n);
                fe_face_values_local.reinit (loc_cell, face_n);


                for (unsigned int q=0; q<n_face_q_points; ++q)
                {

                    const double JxW = fe_face_values_local.JxW(q);

                    for (unsigned int i=0; i<dofs_local_per_cell; ++i)
                    {
                        const double        phi_i_p = fe_face_values_local[pressure].value (i, q);

                        for (unsigned int j=0; j<dofs_local_per_cell; ++j)
                        {
                            const double phi_j_p  = fe_face_values_local[pressure].value (j, q);

                            A_matrix(i,j) += beta_stab_p * phi_i_p * phi_j_p * JxW; //beta_p (p,q)

                        }
                    }
                }
            }

            if (globalProblem)
            {
                for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
                {
                    fe_face_values.reinit (cell, face_n);
                    fe_face_values_local.reinit (loc_cell, face_n);

                    
                    for (unsigned int q=0; q<n_face_q_points; ++q)
                    {
                        
                        const double JxW = fe_face_values_local.JxW(q);
                        const Tensor<1,dim> normal = fe_face_values.normal_vector(q);

                        for (unsigned int i=0; i<dofs_local_per_cell; ++i)
                        {
                            const Tensor<1,dim> phi_i_u = fe_face_values_local[velocities].value (i, q);
                            const double        phi_i_p = fe_face_values_local[pressure].value (i, q);

                            for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
                            {
                                const double  phi_j_m = fe_face_values.shape_value(j, q);

                                B_matrix(i,j) += phi_j_m*(phi_i_u*normal)*JxW;        //  (lamb,v.n)
                                B_matrix(i,j) -= beta_stab_p * phi_i_p * phi_j_m*JxW; // - beta_p (lamb,q)

                            }
                        }
                        for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
                        {
                            const double phi_i_m = fe_face_values.shape_value(i, q);

                            for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
                            {
                                const double phi_j_m = fe_face_values.shape_value(j, q);
                                
                                cell_matrix(i,j) -= beta_stab_p * phi_i_m * phi_j_m * JxW;   //beta_p (lamb,mu)
                                
                            }
                        }
                    }
                }
            }
            else
            {
                for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
                {
                    fe_face_values.reinit (cell, face_n);
                    fe_face_values_local.reinit (loc_cell, face_n);
                    fe_face_values.get_function_values (solution, mult_values);

                    
                    for (unsigned int q=0; q<n_face_q_points; ++q)
                    {
                        
                        const double JxW = fe_face_values.JxW(q);
                        const Tensor<1,dim> normal = fe_face_values.normal_vector(q);

                        for (unsigned int i=0; i<dofs_local_per_cell; ++i)
                        {
                            const Tensor<1,dim> phi_i_u  = fe_face_values_local[velocities].value (i, q);
                            const double        phi_i_p  = fe_face_values_local[pressure].value (i, q);

                            F_vector(i) -= (phi_i_u*normal)*mult_values[q]*JxW;
                            F_vector(i) += beta_stab_p * phi_i_p * mult_values[q] * JxW;
                        }
                    }
                }
            }

          //  A_matrix.print(std::cout,8);

            A_matrix.gauss_jordan();

            if (globalProblem)
            {
                BT_matrix.copy_transposed(B_matrix);
                A_matrix.vmult(aux_vector, F_vector, false);            //  A^{-1} * F
                BT_matrix.vmult(cell_vector, aux_vector, true);        //  B.T * A^{-1} * F - G
                A_matrix.mmult(aux_matrix, B_matrix, false);            //  A^{-1} * B
                BT_matrix.mmult(cell_matrix, aux_matrix, true);         // -C + B.T * A^{-1} * B

                cell->get_dof_indices(dof_indices);

                constraints.distribute_local_to_global (cell_matrix,
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


    void solve ()
    {
        std::cout << "Solving linear system... " << std::endl;
        Timer timer;

        PreconditionJacobi<SparseMatrix<double> > precondition;
        precondition.initialize (system_matrix, .6);

        SolverControl solver_control (system_matrix.m()*10, 1e-11);
        SolverCG<> solver (solver_control);
        solver.solve (system_matrix, solution, system_rhs, PreconditionIdentity() );

        std::cout << std::endl << "   Number of CG iterations: " << solver_control.last_step()
                  << std::endl
                  << std::endl;

        timer.stop();
        std::cout  << "done (" << timer.cpu_time() << "s)" << std::endl;

        constraints.distribute(solution);
    }


    void compute_errors (ConvergenceTable &convergence_table)
    {
        const ComponentSelectFunction<dim>
        pressure_mask (dim, dim+1);
        const ComponentSelectFunction<dim>
        velocity_mask(std::make_pair(0, dim), dim+1);
        
        ExactSolution<dim> exact_solution;
        
        Vector<double> cellwise_errors (triangulation.n_active_cells());
        
        QGauss<dim> quadrature (degree+2);
        
        double u_l2_error, p_l2_error;
        
        VectorTools::integrate_difference (dof_handler_local, solution_local, exact_solution,
                                           cellwise_errors, quadrature,
                                           VectorTools::L2_norm,
                                           &pressure_mask);
        
        p_l2_error = VectorTools::compute_global_error(triangulation,
                                                       cellwise_errors,
                                                       VectorTools::L2_norm);
        
        
        VectorTools::integrate_difference (dof_handler_local, solution_local, exact_solution,
                                           cellwise_errors, quadrature,
                                           VectorTools::L2_norm,
                                           &velocity_mask);
        
        
        u_l2_error = VectorTools::compute_global_error(triangulation,
                                                       cellwise_errors,
                                                       VectorTools::L2_norm);
        
        std::cout << "Errors: ||e_u||_L2 = " << u_l2_error
        << ",   ||e_p||_L2 = " << p_l2_error
        << std::endl
        << std::endl;
        
        convergence_table.add_value("cells", triangulation.n_active_cells());
        convergence_table.add_value("L2_u", u_l2_error);
        convergence_table.add_value("L2_p", p_l2_error);
        
    }


    void output_results () const
    {
        std::string   filename = "solution_local.vtu";
        std::ofstream output (filename.c_str());

        DataOut<dim> data_out;
        std::vector<std::string> names (dim, "Velocity");
        names.emplace_back("Pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> component_interpretation
            (dim+1, DataComponentInterpretation::component_is_part_of_vector);
        component_interpretation[dim] = DataComponentInterpretation::component_is_scalar;
        data_out.add_data_vector (dof_handler_local, solution_local, names, component_interpretation);

        data_out.build_patches (fe_local.degree);
        data_out.write_vtu (output);


        std::string   filename_face = "multiplier.vtu";
        std::ofstream face_output (filename_face.c_str());

        DataOutFaces<dim> data_out_face(false);
        std::vector<std::string> face_name(1,"Trace_pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        face_component_type(1, DataComponentInterpretation::component_is_scalar);

        data_out_face.add_data_vector (dof_handler,
                                       solution,
                                       face_name,
                                       face_component_type);

        data_out_face.build_patches (fe.degree);
        data_out_face.write_vtu (face_output);
    }

};


}


int main ()
{
    using namespace dealii;
    using namespace AulasDealII;

    const int dim = 3;

    ConvergenceTable convergence_table;

    for (int i = 1; i < 6; ++i)
    {
        MixedLaplaceProblem<dim> mixed_laplace_problem(2);
        mixed_laplace_problem.run (i, convergence_table);
    }

    convergence_table.set_precision("L2_u", 2);
    convergence_table.set_scientific("L2_u", true);
    convergence_table.evaluate_convergence_rates("L2_u", "cells", ConvergenceTable::reduction_rate_log2, dim);

    convergence_table.set_precision("L2_p", 2 );
    convergence_table.set_scientific("L2_p", true);
    convergence_table.evaluate_convergence_rates("L2_p", "cells", ConvergenceTable::reduction_rate_log2, dim);

    convergence_table.write_text(std::cout);

    std::ofstream data_output("taxas.dat");
    convergence_table.write_text(data_output);

    std::ofstream tex_output("taxas.tex");
    convergence_table.write_tex(tex_output);

    return 0;
}
